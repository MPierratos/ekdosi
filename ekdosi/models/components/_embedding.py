from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        lookup_keys: List[int],
        num_oov_buckets: int = 1,
        lookups_per_sample: int = 1,
        output_dimension: int = 64,
        random_mask: float = None,
        embedding_kwargs: Dict[str, Any] = dict(),
        **kwargs: Dict[str, Any],
    ) -> None:
        super(EmbeddingLayer, self).__init__()
        self.lookup_keys = lookup_keys
        self.num_oov_buckets = num_oov_buckets
        self.output_dim = output_dimension
        self.extra_args = kwargs
        self.lookup_cardinality = len(lookup_keys) + num_oov_buckets
        self.lookups_per_sample = lookups_per_sample
        self.embedding_kwargs = embedding_kwargs
        if random_mask is not None:
            assert num_oov_buckets == 1
            assert 0.0 <= random_mask <= 1.0
        self.random_mask = random_mask
        self.embedding = nn.Embedding(
            self.lookup_cardinality, self.output_dim, **self.embedding_kwargs
        )
        self.reshaper = nn.Flatten()

    def forward(
        self, inputs: torch.Tensor, return_lookup: bool = False
    ) -> torch.Tensor:
        x = inputs
        if self.training and self.random_mask is not None:
            mask = torch.rand_like(x, dtype=torch.float) < self.random_mask
            x = torch.where(mask, torch.full_like(x, self.lookup_cardinality - 1), x)
        if return_lookup:
            return x
        x = self.embedding(x)
        x = self.reshaper(x)
        return x


class EncoderEmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab: List[str],
        num_oov_buckets: int = 1,
        lookups_per_sample: int = 1,
        output_dimension: int = 64,
        random_mask: float = None,
        embedding_kwargs: Dict[str, Any] = dict(),
        **kwargs: Dict[str, Any],
    ) -> None:
        super(EncoderEmbeddingLayer, self).__init__()
        self.word2idx = {word: ind for ind, word in enumerate(vocab)}
        self.num_oov_buckets = num_oov_buckets
        self.output_dim = output_dimension
        self.extra_args = kwargs
        self.lookup_cardinality = len(vocab) + num_oov_buckets
        self.lookups_per_sample = lookups_per_sample
        self.embedding_kwargs = embedding_kwargs
        if random_mask is not None:
            assert num_oov_buckets == 1
            assert 0.0 <= random_mask <= 1.0
        self.random_mask = random_mask
        self.embedding = nn.Embedding(
            self.lookup_cardinality, self.output_dim, **self.embedding_kwargs
        )
        self.reshaper = nn.Flatten()

    def forward(
        self, text_input: List[str], return_lookup: bool = False
    ) -> torch.Tensor:
        encoded_input = [self.word2idx[word] for word in text_input]
        inputs = torch.tensor(encoded_input)
        x = inputs
        if self.training and self.random_mask is not None:
            mask = torch.rand_like(x, dtype=torch.float) < self.random_mask
            x = torch.where(mask, torch.full_like(x, self.lookup_cardinality - 1), x)
        if return_lookup:
            return x
        x = self.embedding(x)
        x = self.reshaper(x)
        return x


def test_embedding() -> torch.Tensor:
    embedding_layer = EmbeddingLayer(lookup_keys=[0, 1], random_mask=1 / 20)
    input_tensor = torch.tensor([0, 1] * 500)
    output = embedding_layer(input_tensor)
    return output


def test_with_encoder() -> torch.Tensor:
    vocab = ["GRE", "LEW"]
    word2idx = {word: ind for ind, word in enumerate(vocab)}
    text_input = np.array(["GRE", "GRE", "LEW", "GRE", "GRE", "LEW"])
    encoded_input = [word2idx[word] for word in text_input]
    input_tensor = torch.tensor(encoded_input)
    unique_values = torch.unique(input_tensor).tolist()

    embedding_layer = EmbeddingLayer(lookup_keys=unique_values, random_mask=1 / 20)
    output = embedding_layer(input_tensor)
    return output


def test_with_encoder_embedder() -> torch.Tensor:
    vocab = ["GRE", "LEW"]
    text_input = ["GRE", "GRE", "LEW", "GRE", "GRE", "LEW"]
    encoder_embedder = EncoderEmbeddingLayer(vocab, random_mask=1 / 20)
    output = encoder_embedder(text_input)
    return output


def test_embedding_training_inference() -> None:
    embedding_layer = EmbeddingLayer(lookup_keys=[0, 1], random_mask=1.0)
    input_tensor = torch.tensor([0, 1] * 10)

    # Test for training mode, 100% masking applied
    embedding_layer.train()
    lookups = embedding_layer(input_tensor, return_lookup=True)
    assert torch.all(lookups == embedding_layer.lookup_cardinality - 1)
    print(lookups)

    # Test for inference mode, no masking applied
    embedding_layer.eval()
    lookups = embedding_layer(input_tensor, return_lookup=True)
    assert torch.all(lookups != embedding_layer.lookup_cardinality - 1)
    print(lookups)
