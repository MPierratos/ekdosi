
from typing import Dict, Any
import pathlib
import json
from pydantic import BaseModel, Field
from omegaconf import OmegaConf

__all__ = ["TokenizerConfig", "ModelConfig", "ExecutorConfig"]

class TokenizerConfig(BaseModel):
    """Config for a tokenizer.

    Args:
        tokenizer_path (str): Path or name of the tokenizer (local or on huggingface)
        cache_dir (pathlib.Path): location to extract/save hf tokenizer assets 
        padding_side (str): side to add padding, left or right
        trunction_side (str): side to truncate for longer text
        extra_configs (Dict[str, Any]): additional configs to be passed to the tokenizer
    """

    tokenizer_path: str
    cache_dir: pathlib.Path
    padding_side: str = "left"
    truncation_side: str = "right"
    extra_configs: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

class ModelConfig(BaseModel):
    """Config for a model.
    
    Args:
        mod_path (str): Path or name of the model (local or on huggingface)
        cache_dir (pathlib.Path): location to extract/save hf model assets 
        num_layers_unfrozen (str): number of layers to unfreeze for fine-tuning
        peft_config (Any): config for parameter efficient Fine-Tuning library.
                    Peft is used to reduce the number of parameters to train.
                    (i.e. https://github.com/huggingface/peft)
            Example config for LORA:
                {"peft_type": "LORA", 
                  "task_type"
                  "r": 8, 
                  "lora_alpha":32, 
                  "lora_dropout":0.05,
                  "task_type":"QUESTION_ANS"
                  "inference_mode":False,
                  "fan_in_fan_out"=False,
                  "bias":"none"
                  "target_modules":["q_lin", "k_lin", "v_lin", "out_lin"]
                  }
            Note: to fetch target_modules, look at print(model) and look at Attention layers
    
        extra_configs (Dict[str, Any]): additional configs to be passed to the tokenizer
    """

    mod_path: str
    cache_dir: pathlib.Path
    num_layers_unfrozen: int = -1
    peft_config: Any = None
    extra_configs: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


class ExecutorConfig(BaseModel):
    """Top level config for a pipeline"""

    model: ModelConfig
    #optimizer: OptimizerConfig
    #scheduler: SchedulerConfig
    tokenizer: TokenizerConfig
    #train: TrainConfig

    @classmethod
    def load_yaml(cls, yaml_path: pathlib.Path|str):
        """Load yaml file as ExecutorConfig.

        Args:
            yaml_path (pathlib.Path | str): path to the config file

        """
        config = OmegaConf.load(yaml_path)
        config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
        return cls.from_dict(config)
    
    def to_dict(self):
        
        data = {
            "tokenizer": self.tokenizer.__dict__,
            "model": self.model.__dict__,
        }

        return data

    @classmethod
    def from_dict(cls, config: Dict):
        return cls(
            tokenizer=TokenizerConfig.from_dict(config.tokenizer),
            model=ModelConfig.from_dict(config.model),
        )
   

    def __str__(self) -> str:
        """Returns a human-readable string representation of the config."""
        return json.dumps(self.to_dict(), indent=4)
  