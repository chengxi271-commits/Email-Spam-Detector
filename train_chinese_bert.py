# train_chinese_bert.py (Simplified to load preprocessed .pt files)

import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据集类
class EnhancedDataset(Dataset):
    def __init__(self, data_dict):
        self.encodings = data_dict['encodings']
        self.labels = data_dict['labels']
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# --- 路径定义 ---
TRAIN_DATA_FILE = './data/train_enhanced_dataset.pt'
VAL_DATA_FILE = './data/validation_enhanced_dataset.pt'
MODEL_SAVE_PATH = './saved_models/chinese_bert_enhanced_v1'
PRETRAINED_MODEL_PATH = './local_models/hfl-bert-wwm'

# --- 直接加载预处理好的数据 ---
logging.info("Loading preprocessed datasets...")
# 增加 weights_only=False 参数
train_data_dict = torch.load(TRAIN_DATA_FILE, weights_only=False)
val_data_dict = torch.load(VAL_DATA_FILE, weights_only=False)

train_dataset = EnhancedDataset(train_data_dict)
val_dataset = EnhancedDataset(val_data_dict)

logging.info("Preprocessed datasets loaded successfully!")

# --- 加载模型和定义训练参数 ---
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

training_args = TrainingArguments(
    output_dir='./results_chinesebert',
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=48,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)

def compute_metrics(p):
    pred, labels = p
    pred = pred.argmax(axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- 开始训练 ---
logging.info("Starting model fine-tuning...")
trainer.train()
logging.info("Training finished.")

# --- 保存模型 ---
logging.info(f"Saving best model and tokenizer to {MODEL_SAVE_PATH}...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
logging.info("Process complete!")