import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.dataset import CommonLitDataset
from src.train import CommonLitBertModel


@pytest.fixture
def sample_dataset():
    data = pd.DataFrame()
    data["excerpt"] = [
        "Hello world!",
        "What is the name of the repository ?",
        "This library is not a modular toolbox of building blocks for neural nets.",
    ]
    data["target"] = np.random.rand(3, 1)

    model_path = "bert-base-uncased"
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    dataset = CommonLitDataset(data, tokenizer=tokenizer, max_len=100)
    return dataset


def test_bert_model(sample_dataset):
    batch_size = 1
    dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    first_batch = iter(dataloader).next()

    model = CommonLitBertModel()
    z = model(first_batch)

    assert first_batch["target"].shape == z.shape
