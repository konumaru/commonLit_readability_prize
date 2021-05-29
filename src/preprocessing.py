import numpy as np
import pandas as pd
import torch
from transformers import (
    AdamW,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    print(last_hidden_states)


if __name__ == "__main__":
    main()
