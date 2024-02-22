from ekdosi.models.components import EncoderEmbeddingLayer
from torch import nn
import torch
import math
from torch.nn import MultiheadAttention

__all__ = ["GeometricStepDownDenseLayer", "ModelAttention"]

def generate_geometric_series(min_n: int, max_n: int, reverse: bool = False) -> list[int]:
    """
    Generates a geometric series from min_n to max_n.
    """
    series = [int(2**i) for i in range(min_n, max_n + 1)]
    return series[::-1] if reverse else series

class GeometricStepDownDenseLayer(nn.Module):
    """
    Creates a dense set of layers that steps down gradually.
    """
    def __init__(self, input_dim: int, min_n: int = 5, dropout: float = 0.05, batch_norm: bool = True):
        super().__init__()
        max_n = math.floor(math.log2(input_dim))
        series = generate_geometric_series(min_n, max_n, reverse=True)
        self.output_dim = series[-1]
        self.layers = nn.ModuleList()
        for units in series:
            self.layers.extend([
                nn.Linear(input_dim, units),
                nn.Dropout(p=dropout),
                nn.BatchNorm1d(units) if batch_norm else nn.Identity()
            ])
            input_dim = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer.
    """
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma if self.training else x

class ModelAttention(nn.Module):
    def __init__(self, input_size: int, embedding_vocab: list[str], output_size: int, feature_self_attn_num_heads: int = 8, feature_cross_attn_num_heads: int = 1):
        super().__init__()
        self.output_size = output_size
        self.gaussian_noise = GaussianNoise(sigma=0.01)
        self.embed = EncoderEmbeddingLayer(embedding_vocab, random_mask=1/20)
        
        self.new_embed_dim = (input_size // feature_self_attn_num_heads + 1) * feature_self_attn_num_heads if input_size % feature_self_attn_num_heads != 0 else input_size
        self.adjust_dim_layer = nn.Linear(input_size, self.new_embed_dim) if input_size % feature_self_attn_num_heads != 0 else nn.Identity()

        self.multihead_attn = MultiheadAttention(embed_dim=self.new_embed_dim, num_heads=feature_self_attn_num_heads, batch_first=True)
        self.embedding_projection = nn.Linear(self.embed.output_dim, self.new_embed_dim)
        self.cross_attention = MultiheadAttention(embed_dim=self.new_embed_dim, num_heads=feature_cross_attn_num_heads, batch_first=True)
        
        self.geo_step_features = GeometricStepDownDenseLayer(input_dim=self.new_embed_dim, min_n=5, batch_norm=True)
        output_geo_layer_size = round(math.log2(output_size))
        self.geo_step_final = GeometricStepDownDenseLayer(input_dim=self.geo_step_features.output_dim + self.embed.output_dim, min_n=output_geo_layer_size, batch_norm=True)
        self.final_ll = nn.Linear(self.geo_step_final.output_dim, output_size)

    def forward(self, x, embed):
        features = torch.as_tensor(x, dtype=torch.float32)
        features_w_noise = self.gaussian_noise(features).unsqueeze(1)
        features_w_noise_adj = torch.relu(self.adjust_dim_layer(features_w_noise))
        
        feature_attn_output, _ = self.multihead_attn(features_w_noise_adj, features_w_noise_adj, features_w_noise_adj)
        feature_attn_output = feature_attn_output.squeeze(1)
        
        embedding = self.embed(embed)
        embed_projection = torch.relu(self.embedding_projection(embedding))
        features_cross_embedding, _ = self.cross_attention(embed_projection.unsqueeze(1), feature_attn_output.unsqueeze(1), feature_attn_output.unsqueeze(1))
        features_cross_embedding = features_cross_embedding.squeeze(1)
        
        features_stepdown = self.geo_step_features(features_cross_embedding)
        concatenated = torch.cat((features_stepdown, embedding), dim=1)
        x = torch.relu(self.final_ll(self.geo_step_final(concatenated)))
        return x

def test_model():

    features = torch.randn(3, 37)  # 3 is the batch size, 37 features
    features_embed = ['material1', 'material2', 'material1']

    embedding_vocab = ['material1', 'material2']

    model = ModelAttention(input_size=features.shape[1],
                                embedding_vocab=embedding_vocab,
                                output_size=7,
                                feature_self_attn_num_heads=8,
                                feature_cross_attn_num_heads=2)

    out = model(features, features_embed)
    print(model)