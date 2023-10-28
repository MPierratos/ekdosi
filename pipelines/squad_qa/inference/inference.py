import timeit

from peft import PeftConfig, PeftModel
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    pipeline,
)

def lora_infer(question, context, MODEL_PATH):
    """Inference with Lora"""
    config = PeftConfig.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(
        DistilBertForQuestionAnswering.from_pretrained(config.base_model_name_or_path),
        MODEL_PATH,
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    start = timeit.default_timer()
    qa_model = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    stop = timeit.default_timer()
    print(f"Inference Time: {stop-start:.2f}s")
    return qa_model(question=question, context=context)


def vanilla_infer(question, context, MODEL_PATH):
    """Inference without Lora"""
    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    start = timeit.default_timer()
    qa_model = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    stop = timeit.default_timer()
    print(f"Inference Time: {stop-start:.2f}s")
    return qa_model(question=question, context=context)


if __name__ == "__main__":
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    EPOCHS = 3

    MODEL = "distilbert-base-uncased"
    MODEL_PATH = f"/mnt/n/projects/squad/models/{MODEL}-lr{LEARNING_RATE}-epochs{EPOCHS}-batchsize{BATCH_SIZE}"
    LORA = True

    question = "Where is the github link?"
    context = "You can find the github link for this video in the description"

    if LORA:
        print(lora_infer(question, context, MODEL_PATH))
    else:
        print(vanilla_infer(question, context, MODEL_PATH))
