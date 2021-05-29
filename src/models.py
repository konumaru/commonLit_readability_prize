import pytorch_lightning as pl
import pytorch_warmup as warmup
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
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
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
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
    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        output_hidden_states: bool = False,
    ):
        super(CommonLitRoBERTaModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=output_hidden_states,
        )
        self.config = self.roberta.config
        self.regression_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.5),
            nn.Linear(768, 1),
        )
        # Initialize Weights
        self.regression_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        outputs = self.roberta(**batch["inputs"])
        pooler_output = outputs.pooler_output
        hidden_state_avg = (
            outputs.last_hidden_state[:, -4:].mean(dim=(1, 2)).view(-1, 1)
        )
        hidden_state_sum = outputs.last_hidden_state[:, -4:].sum(dim=(1, 2)).view(-1, 1)

        # x = torch.cat((pooler_output, hidden_state_avg, hidden_state_sum), dim=1)
        x = self.regression_head(pooler_output)
        return x


class CommonLitModel(pl.LightningModule):
    def __init__(
        self,
        num_epoch: int = 20,
        train_dataloader_len: int = 20,
        lr: float = 1e-4,
    ):
        super(CommonLitModel, self).__init__()
        self.lr = lr
        self.num_epoch = num_epoch
        self.train_dataloader_len = train_dataloader_len

        self.model = CommonLitRoBERTaModel()
        self.loss_fn = RMSELoss()  # nn.MSELoss()

    def forward(self, batch):
        z = self.model(batch)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=5e-2,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=self.train_dataloader_len * self.num_epoch,
        )
        return [optimizer], [lr_scheduler]

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
