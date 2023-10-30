
import importlib


if importlib.util.find_spec("peft") is not None:
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_config,
        get_peft_model,
        prepare_model_for_int8_training
    )


