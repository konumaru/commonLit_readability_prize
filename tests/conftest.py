import warnings

warnings.simplefilter("ignore")

import numpy as np
import pytest
import torch
from transformers import BertModel, BertTokenizer

from src.dataset import CommonLitDataset


@pytest.fixture
def sample_dataset():
    sample_text = [
        "Hello world!",
        "What is the name of the repository ?",
        "This library is not a modular toolbox of building blocks for neural nets.",
    ]

    target = np.random.rand(1, len(sample_text))
    model_path = "bert-base-uncased"

    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    dataset = CommonLitDataset(
        target, excerpt=sample_text, tokenizer=tokenizer, max_len=100
    )
    return dataset
