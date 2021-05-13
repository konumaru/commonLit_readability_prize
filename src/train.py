import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


class CommonLitBertModel(nn.Module):
    def __init__(self):
        super(CommonLitBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self, batch):
        ids, mask, token_type_ids = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        )
        _, output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        output = self.out(output)
        return output


def main():
    pass


if __name__ == "__main__":
    main
