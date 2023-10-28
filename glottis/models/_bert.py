import logging

import torch
import torch.nn as nn
from glottis.models import register_model

logger = logging.getLogger(__name__)


class BERTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_segments: int,
        max_len: int,
        embed_dim: int,
        dropout: float,
    ):
        """Initialize bert embedding, this includes three embeddings:

        token embedding, segment embedding, and position embedding

        Args:
            vocab_size (int): vocabulary size, i.e. WordPiece embedding has 30k tokens
            n_segments (int): segments, i.e. in Bert you train 2 segments at a time, plus padding
            max_len (int): combined number of tokens for sentence A and sentence B
            embed_dim (int): embedding dimension, i.e. hidden size
            dropout (float): dropout probability to use across all layers
        """
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)

        # initialize embedding for positional embedding
        self.pos_inp = torch.tensor([i for i in range(max_len)])

    def forward(self, seq, seg):
        embed_val = (
            self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        )
        return embed_val

@register_model
class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30_000,
        n_segments: int = 3,
        max_len: int = 512,
        embed_dim: int = 512,
        n_layers: int = 12,
        attn_heads: int = 12,
        dropout: float = 0.1,
    ):
        """Implementation of Bert Base

        Args:
            vocab_size (int): vocabulary size, i.e. WordPiece embedding has 30k tokens
            n_segments (int): segments, i.e. in Bert you train 2 segments at a time, plus padding
            max_len (int): combined number of tokens for sentence A and sentence B
            embed_dim (int): embedding dimension, i.e. hidden size
            n_layers (int): number of transformer blocks
            attn_heads (int): number of attention heads
            dropout (float): dropout probability to use across all layers
        """

        super().__init__()

        self.embedding = BERTEmbedding(
            vocab_size, n_segments, max_len, embed_dim, dropout
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, attn_heads, embed_dim * 4
        )
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out


def test_bert():
    VOCAB_SIZE = 30_000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1

    sample_seq = torch.randint(
        high=VOCAB_SIZE,
        size=[
            MAX_LEN,
        ],
    )
    sample_seg = torch.randint(
        high=N_SEGMENTS,
        size=[
            MAX_LEN,
        ],
    )

    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding_tensor = embedding(sample_seq, sample_seg)
    logger.info(f"Embedding size: {embedding_tensor.size()}")

    bert = BERT(
        VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT
    )
    out = bert(sample_seq, sample_seg)
    logger.info(f"BERT output: {out.size()}")

    return out
