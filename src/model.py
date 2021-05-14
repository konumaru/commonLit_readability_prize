import pytorch_lightning as pl
import pytorch_warmup as warmup
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
    RobertaModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


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


class CommonLitRoBERTaModel(nn.Module):
    def __init__(self):
        super(CommonLitRoBERTaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(768, 1)

    def forward(self, batch):
        ids, mask = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        _, output = self.roberta(
            ids,
            attention_mask=mask,
            return_dict=False,
        )
        output = self.fc(output)
        return output


class CommonLitModel(pl.LightningModule):
    def __init__(self, base_model, num_epoch, train_dataloader_len, lr=1e-4):
        super(CommonLitModel, self).__init__()
        self.lr = lr
        self.num_epoch = num_epoch
        self.train_dataloader_len = train_dataloader_len

        self.model = base_model
        self.loss_fn = RMSELoss()  # nn.MSELoss()

    def forward(self, batch):
        z = self.model(batch)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-2,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=self.train_dataloader_len * self.num_epoch,
        )
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop = EarlyStopping(
            mode="min",
            patience=20,
            verbose=False,
            monitor="val_loss",
            min_delta=0.0,
        )
        checkpoint = ModelCheckpoint(
            filename="{epoch:02d}-{loss:.4f}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
        )
        return [checkpoint, lr_monitor, early_stop]

    def shared_step(self, batch):
        z = self(batch)
        loss = self.loss_fn(z, batch["target"])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return {"test_loss": loss}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        loss = torch.cat([out["test_loss"] for out in outputs], dim=0)
        self.log("test_rmse", torch.mean(loss))
