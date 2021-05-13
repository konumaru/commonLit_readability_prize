import warnings

warnings.simplefilter("ignore")

import numpy as np
import pytest
import torch
from transformers import BertModel, BertTokenizer

from src.dataset import CommonLitDataset


@pytest.fixture
def some_text_data():
    text = [
        "Hello world!",
        "What is the name of the repository ?",
        "This library is not a modular toolbox of building blocks for neural nets.",
    ]
    return text


def test_dataset(some_text_data):
    target = np.random.rand(1, len(some_text_data))
    model_path = "bert-base-uncased"

    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    dataset = CommonLitDataset(
        target, excerpt=some_text_data, tokenizer=tokenizer, max_len=100
    )
