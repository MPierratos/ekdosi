import dataclasses as dc
import logging
import pathlib
from typing import List

import torch
import transformers
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from datasets import DATA_DIR, SquadDataset
from ekdosi.configs import ExecutorConfig
from ekdosi.utils import get_optimizer

logger = logging.getLogger("ekdosi")


@dc.dataclass
class Trainer:
    name: str
    config: ExecutorConfig

    def __post_init__(self):
        # ignore verbosity errors in transformers library
        transformers.utils.logging.set_verbosity_error()

        self.tokenizer = self.setup_tokenizer()

        self.model = self.setup_model()

        self.optimizer = self.setup_optimizer()

    def setup_tokenizer(self) -> AutoTokenizer:
        """Setup the tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer.tokenizer_path,
            **self.config.tokenizer.tokenizer_extra_configs,
        )
        tokenizer.padding_side = self.config.tokenizer.padding_side
        tokenizer.truncation_side = self.config.tokenizer.truncation_side
        return tokenizer

    def setup_model(self) -> AutoModelForQuestionAnswering:
        """Setup the model architeecture.

        Currently calls a QA model.

        # TODO: expand to multiple model types
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.model.mod_path, **self.config.model.mod_extra_configs
        ).to_device(device)

        if "peft_config" in config.model:
            if config.model.peft_config["peft_type"] == "LORA":
                peft_config = LoraConfig(**config.model.peft_config)

                logger.info("#" * 32, "Trainable parameters Before LoRA", "#" * 32)
                logger.info(model.num_parameters())
                model = get_peft_model(model, peft_config)
                logger.info("#" * 32, "Trainable parameters After LoRA", "#" * 32)
                model.print_trainable_parameters()

            else:
                raise NotImplementedError("Only peft_type of LORA exists currently.")
        return model

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer"""
        if hasattr(self, "model"):
            optimizer_class = get_optimizer(self.config.optimizer)
            optimizer = optimizer_class(
                self.model_parameters, **self.config.optimizer.optimizer_extra_configs
            )
            return optimizer
        else:
            raise ValueError("No model found.")

    def train(
        self,
        dataset: Dataset,
        data_split: List[float] = [0.8, 0.1, 0.1],
        debug_mode: bool = False,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if debug_mode:
            data_sample_size = 10_000

        BATCH_SIZE = config.train.batch_size

        data = dataset(
            data_path=DATA_PATH,
            data_sample_size=data_sample_size,
            tokenizer=self.tokenizer,
        )

        generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset, test_dataset = random_split(
            data, data_split, generator=generator
        )

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        for epoch in tqdm(range(config.train.epochs)):
            self.model.train()
            train_running_loss = 0
            for idx, sample in enumerate(tqdm(train_dataloader)):
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)
                start_positions = sample["start_positions"].to(device)
                end_positions = sample["end_positions"].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()

            train_loss = train_running_loss / (idx + 1)

            self.model.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, sample in enumerate(tqdm(val_dataloader)):
                    input_ids = sample["input_ids"].to(device)
                    attention_mask = sample["attention_mask"].to(device)
                    start_positions = sample["start_positions"].to(device)
                    end_positions = sample["end_positions"].to(device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )

                    val_running_loss += outputs.loss.item()
                val_loss = val_running_loss / (idx + 1)

            logger.info("-" * 30)
            logger.info(f"Train Loss Epoch {epoch+1}: {train_loss:.4f}")
            logger.info(f"Valid Loss Epoch {epoch+1}: {val_loss:.4f}")
            logger.info("-" * 30)

        MODEL_SAVE_PATH = f"/mnt/n/projects/squad/models/{self.name}"

        self.model.save_pretrained(MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(MODEL_SAVE_PATH)

        torch.cuda.empty_cache()

        self.model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_dataloader)):
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)
                start_positions = sample["start_positions"]
                end_positions = sample["end_positions"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                start_pred = torch.argmax(outputs["start_logits"], dim=1).cpu().detach()
                end_pred = torch.argmax(outputs["end_logits"], dim=1).cpu().detach()

                preds.extend([[int(i), int(j)] for i, j in zip(start_pred, end_pred)])
                true.extend(
                    [[int(i), int(j)] for i, j in zip(start_positions, end_positions)]
                )

        preds = [item for sublist in preds for item in sublist]
        true = [item for sublist in true for item in sublist]

        f1_value = f1_score(true, preds, average="macro")
        logger.info(f1_value)


if __name__ == "__main__":
    config_path = pathlib.Path(__file__).parent.parent / "config.yml"
    logger.info(config_path)
    config = ExecutorConfig.load_yaml(config_path)

    qa_train = Trainer("distilbert-tmp", config)

    DATA_PATH = DATA_DIR / "squad/data/raw/train-v2.0.json"

    qa_train.train(dataset=SquadDataset, debug_mode=True)
