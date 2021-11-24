from typing import Optional

import torch
import datasets
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', batch_size=32):
        super().__init__()
        
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def prepare_data(self):
        cola_dataset = load_dataset('glue', 'cola')
        self.train_dataset = cola_dataset['train']
        self.val_dataset = cola_dataset['validation']
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.train_dataset.map(self.tokenize_data, batched=True)
            self.train_dataset.set_format(
                type="torch", columns=['input_ids', 'attention_mask', 'label']
            )
            
            self.val_dataset = self.val_dataset.map(self.tokenize_data, batched=True)
            self.val_dataset.set_format(
                type="torch", columns=['input_ids', 'attention_mask', 'label']
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        

if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))['input_ids'].shape)
