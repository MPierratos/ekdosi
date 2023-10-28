import logging

import torch
import transformers
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

from glottis.datasets import SquadDataset

from glottis.configs import ExecutorConfig

import pathlib

config_path = pathlib.Path(__file__).parent / "config.yml"
config  = ExecutorConfig.load_yaml()


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    EPOCHS = 3
    DATA_PATH = "/mnt/n/projects/squad/data/raw/train-v2.0.json"
    CACHE_DIR = "/mnt/n/projects/.cache/"

    MODEL_PATH = "distilbert-base-uncased"
    MODEL_SAVE_PATH = f"/mnt/n/projects/squad/models/{MODEL_PATH}-lr{LEARNING_RATE}-epochs{EPOCHS}-batchsize{BATCH_SIZE}/"
    LORA = True

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    model = DistilBertForQuestionAnswering.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR
    ).to(device)

    if LORA:
        config = LoraConfig(
            task_type="QUESTION_ANS",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            fan_in_fan_out=False,
            bias="none",
            # to fetch target_modules, look at print(model) and look at Attention layers
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        )
        logger.info("#" * 32, "Trainable parameters Before LoRA", "#" * 32)
        logger.info(model.num_parameters()) 
        model = get_peft_model(model, config)
        logger.info("#" * 32, "Trainable parameters After LoRA", "#" * 32)
        model.print_trainable_parameters()

    dataset = SquadDataset(
        data_path=DATA_PATH, data_sample_size=20_000, tokenizer=tokenizer
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator
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

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, sample in enumerate(tqdm(train_dataloader)):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            start_positions = sample["start_positions"].to(device)
            end_positions = sample["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(val_dataloader)):
                input_ids = sample["input_ids"].to(device)
                attention_mask = sample["attention_mask"].to(device)
                start_positions = sample["start_positions"].to(device)
                end_positions = sample["end_positions"].to(device)

                outputs = model(
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

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    torch.cuda.empty_cache()

    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_dataloader)):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            start_positions = sample["start_positions"]
            end_positions = sample["end_positions"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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
