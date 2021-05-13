import pytorch_lightning as pl
import torch.nn as nn
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
        self.fc = nn.Linear(768, 1)

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
        output = self.fc(output)
        return output


class CommonLitModel(pl.LightningModule):
    def __init__(self):
        super(CommonLitModel, self).__init__()
        self.model = CommonLitBertModel()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        z = self.model(batch)
        return z

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def configure_callbacks(self):
        # early_stop = EarlyStopping(monitor="val_acc", mode="max")
        # checkpoint = ModelCheckpoint(monitor="val_loss")
        # return [early_stop, checkpoint]
        pass

    def shared_step(self, batch):
        z = self(batch)
        loss = self.loss_fn(z, batch["target"])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return {"test_loss": loss}
