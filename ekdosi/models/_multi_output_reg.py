import time

import h5py
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf  # Use OmegaConf instead of hydra
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

from ekdosi.configs.trainer.config import ExecutorConfig
from ekdosi.models.components import EncoderEmbeddingLayer, GeometricStepDownDenseLayer


def print_progress(start_time, progress_percentage):
    elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
    estimated_time = elapsed_time / progress_percentage - elapsed_time
    print(
        f"Elapsed time: {elapsed_time:.2f}m, Estimated time remaining: {estimated_time:.2f}m"
    )


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as file:
            self.features = file["train"]["features"][:]
            self.embeddings = file["train"]["embeddings"][:]
            self.targets = file["train"]["targets"][:]
            self.embedding_fields = file["train"]["embedding_fields"][:]
        self.length = self.features.shape[0]
        self.output_shape = self.targets.shape[
            -1
        ]  # Extract the output shape from the targets

    def __getitem__(self, index):
        features = self.features[index]
        embeddings = self.embeddings[index]
        targets = self.targets[index]
        return (features, embeddings), targets

    def __len__(self):
        return self.length


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, embedding_vocab, output_size):
        super(SimpleNet, self).__init__()
        self.encoder_embedding = EncoderEmbeddingLayer(vocab=embedding_vocab)
        self.fc1 = nn.Linear(input_size + self.encoder_embedding.output_dim, 64)
        self.fc2 = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_size)])

    def forward(self, x):
        features, embeddings = x
        embeddings = self.encoder_embedding(embeddings)
        x = torch.cat((features, embeddings), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.cat([fc(x) for fc in self.fc2], dim=1)
        return x


class Trainer:
    def __init__(self, config: ExecutorConfig):
        """
        Initializes the Trainer class.

        Args:
            config (Config): The configuration settings.
        """
        self.config = config
        self.train_loader, self.val_loader = self._get_data_loaders(config.dataset.name)
        self.epochs = config.epochs
        self.learning_rate = config.optimizer.optimizer_extra_configs.lr
        self.model = self._get_model(config.model.name)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = None
        self._setup_scheduler(config)
        self._setup_loss_function()
        self.writer = SummaryWriter()
        self.train_losses = []
        self.val_losses = []

    def _get_data_loaders(self, dataset_name):
        dataset_file_path = (
            f"{dataset_name}.hdf5"  # Construct the file path from the dataset name
        )
        self.dataset = HDF5Dataset(dataset_file_path)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config.train.split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(
            self.dataset, batch_size=self.config.train.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(
            self.dataset, batch_size=self.config.train.batch_size, sampler=val_sampler
        )
        return train_loader, val_loader

    def _get_model(self, model_name):
        if model_name == "SimpleNet":
            return SimpleNet(
                self.dataset.features.shape[1],
                self.dataset.embedding_fields,
                self.dataset.output_shape,
            )  # Use output_shape from dataset
        else:
            raise ValueError("Unsupported model")

    def _setup_scheduler(self, config):
        if hasattr(config, "scheduler") and config.scheduler.name not in [None, "None"]:
            if config.scheduler.name == "StepLR":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=config.scheduler.scheduler_extra_configs.step_size,
                    gamma=config.scheduler.scheduler_extra_configs.gamma,
                )
            elif config.scheduler.name == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.scheduler.scheduler_extra_configs.T_max,
                    eta_min=config.scheduler.scheduler_extra_configs.eta_min,
                )
            else:
                raise ValueError("Unsupported learning rate scheduler")

    def _setup_loss_function(self):
        self.loss_function = nn.MSELoss(reduction="mean")

    def train(self):
        start_time = time.time()
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()  # Step the learning rate scheduler
                running_loss += loss.item()
                if i % 10 == 9:
                    self.writer.add_scalar(
                        "training loss",
                        running_loss / 10,
                        epoch * len(self.train_loader) + i,
                    )
                    running_loss = 0.0
            epoch_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(epoch_train_loss)
            self.writer.add_scalar("epoch training loss", epoch_train_loss, epoch)
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs, labels = data
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()
            epoch_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(epoch_val_loss)
            self.writer.add_scalar("epoch validation loss", epoch_val_loss, epoch)
            if epoch_val_loss < best_val_loss:
                torch.save(self.model.state_dict(), "model.pt")
                best_val_loss = epoch_val_loss
            self.model.train()
            print_progress(start_time, (epoch + 1) / self.epochs)
        self.writer.close()
        print("Finished Training")


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.show()
