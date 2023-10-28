import json
import pathlib

import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    def __init__(self, data_path: pathlib.Path, data_sample_size: int, tokenizer):
        contexts, questions, answers = self._read_data(data_path, data_sample_size)
        answers = self._add_end_idx(contexts, answers)

        # tokenize the context and question together with a SEP in the middle
        encodings = tokenizer(contexts, questions, padding=True, truncation=True)

        self.encodings = self._update_start_end_positions(encodings, answers, tokenizer)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def _read_data(self, path: pathlib.Path, data_sample_size: int = None):
        if data_sample_size is not None:
            assert data_sample_size > 0, "data sample size must be > 0"

        with open(path, "rb") as f:
            squad = json.load(f)

            contexts = []
            questions = []
            answers = []
            for group in squad["data"]:
                for passage in group["paragraphs"]:
                    context = passage["context"]
                    for qa in passage["qas"]:
                        question = qa["question"]
                        for answer in qa["answers"]:
                            contexts.append(context)
                            questions.append(question)
                            answers.append(answer)
        if data_sample_size:
            return (
                contexts[:data_sample_size],
                questions[:data_sample_size],
                answers[:data_sample_size],
            )
        else:
            return contexts, questions, answers

    def _add_end_idx(self, contexts: dict, answers: dict):
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]

            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx - 1 : end_idx - 1] == gold_text:
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx - 2 : end_idx - 2] == gold_text:
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2
        return answers

    def _update_start_end_positions(self, encodings, answers, tokenizer):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(
                encodings.char_to_token(i, answers[i]["answer_start"])
            )
            end_positions.append(
                encodings.char_to_token(i, answers[i]["answer_end"] - 1)
            )
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions

        return encodings


def test_squad_dataset():
    from transformers import DistilBertTokenizerFast
    from datasets import DATA_DIR

    CACHE_DIR = "/mnt/n/projects/.cache/"
    data_path = DATA_DIR / "squad/data/raw/dev-v2.0.json"

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased", cache_dir=CACHE_DIR
    )
    dataset = SquadDataset(
        data_path=data_path, data_sample_size=20_000, tokenizer=tokenizer
    )
    print(dataset[0].keys())
    return dataset
