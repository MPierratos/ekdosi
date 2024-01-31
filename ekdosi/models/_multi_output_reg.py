import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from ekdosi.models.components import GeometricStepDownDenseLayer, EncoderEmbeddingLayer
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            self.features = file['train']['features'][:]
            self.embeddings = file['train']['embeddings'][:]
            self.targets = file['train']['targets'][:]
        self.length = self.features.shape[0]

    def __getitem__(self, index):
        features = self.features[index]
        embeddings = self.embeddings[index]
        targets = self.targets[index]
        return (features, embeddings), targets

    def __len__(self):
        return self.length

# Load the dataset
dataset = HDF5Dataset('your_file.hdf5')

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, embedding_vocab, output_size):
        super(SimpleNet, self).__init__()
        self.encoder_embedding = EncoderEmbeddingLayer(vocab=embedding_vocab)
        self.fc1 = nn.Linear(input_size + self.encoder_embedding.output_dim, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        features, embeddings = x
        embeddings = self.encoder_embedding(embeddings)
        x = torch.cat((features, embeddings), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function and optimizer
embedding_vocab = ['your', 'vocab', 'list']  # Define your embedding vocabulary list
model = SimpleNet(dataset.features.shape[1], embedding_vocab, 3)  # Changed output size to 3
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Changed optimizer to Adam

# Initialize TensorBoard
writer = SummaryWriter()

# Training loop
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Log every 10 mini-batches
            writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    # Log epoch loss
    writer.add_scalar('epoch training loss', running_loss / len(train_loader), epoch)

    # Validation loss
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    writer.add_scalar('epoch validation loss', val_loss / len(val_loader), epoch)
    model.train()

writer.close()  # Close the writer when we're finished using it
print('Finished Training')

