import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import CommonLitDataModule, CommonLitDataset
from src.models import CommonLitModel, CommonLitRoBERTaModel, RMSELoss


@pytest.fixture
def sample_data():
    data = pd.DataFrame()
    data["excerpt"] = [
        "Hello world!",
        "What is the name of the repository ?",
        "This library is not a modular toolbox of building blocks for neural nets.",
    ]
    data["target"] = np.random.rand(3, 1)

    num_textstat_dim = 10
    for i in range(num_textstat_dim):
        data[f"textstats_{i}"] = np.random.rand(3, 1)

    return data


def test_roberta_model(sample_data):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = CommonLitDataset(sample_data, tokenizer)
    batch = iter(DataLoader(dataset, batch_size=2)).next()

    model = CommonLitRoBERTaModel(
        model_name_or_path="roberta-base", output_hidden_states=False
    )

    z = model(batch)
    assert z.shape == batch["target"].shape

    loss_fn = RMSELoss()
    loss = loss_fn(z, batch["target"])
    loss.backward()


def test_pl_model(sample_data):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = CommonLitDataset(sample_data, tokenizer)
    batch = iter(DataLoader(dataset, batch_size=2)).next()

    model = CommonLitModel()

    z = model(batch)
    assert z.shape == batch["target"].shape

    loss_fn = RMSELoss()
    loss = loss_fn(z, batch["target"])
    loss.backward()
