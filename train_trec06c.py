# train_trec06c.py (最终修正版)

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset
import logging

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 定义数据集类
class TrecDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# --- 定义常量 ---
NUM_SHOTS = 50
TRAIN_FILE_PATH = "./data/train.csv"
VALIDATION_FILE_PATH = "./data/validation.csv"
MODEL_SAVE_PATH = f"./saved_models/trec06c_classifier_{NUM_SHOTS}_shot"
MODEL_NAME = "./local_models/hfl-bert-wwm"

# --- 数据加载与采样 ---
logging.info(f"Loading data to create a {NUM_SHOTS}-shot training set...")
try:
    df_train_full = pd.read_csv(TRAIN_FILE_PATH).dropna()
    df_val = pd.read_csv(VALIDATION_FILE_PATH).dropna()
except FileNotFoundError:
    logging.error(f"Data files not found. Please run preprocess_trec06c.py first.")
    exit()

num_per_class = NUM_SHOTS // 2
df_train_spam = df_train_full[df_train_full["label"] == 1].head(num_per_class)
df_train_ham = df_train_full[df_train_full["label"] == 0].head(
    NUM_SHOTS - num_per_class
)
df_train = pd.concat([df_train_spam, df_train_ham]).sample(frac=1, random_state=42)
logging.info(f"Successfully sampled {len(df_train)} training examples.")

train_texts = df_train["text"].astype(str).tolist()
train_labels = df_train["label"].tolist()
val_texts = df_val["text"].astype(str).tolist()
val_labels = df_val["label"].tolist()

# --- 分词 ---
logging.info(f"Loading tokenizer from local path: {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

logging.info(f"Tokenizing the {NUM_SHOTS}-shot training set and the validation set...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

train_dataset = TrecDataset(train_encodings, train_labels)
val_dataset = TrecDataset(val_encodings, val_labels)

# --- 加载模型 ---
logging.info(f"Loading pre-trained model from local path: {MODEL_NAME}...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- 定义训练参数 ---
# ####################################################################
# 关键修正点：将 evaluation_strategy 改为 eval_strategy
# ####################################################################
training_args = TrainingArguments(
    output_dir="./results_few_shot",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    logging_steps=10,
    eval_strategy="steps",  # <-- 已修正
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)


def compute_metrics(p):
    pred, labels = p
    pred = pred.argmax(axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


# --- 初始化训练器并开始训练 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

logging.info(f"Starting {NUM_SHOTS}-shot model fine-tuning...")
trainer.train()
logging.info("Training finished.")

# --- 保存最好的模型 ---
logging.info(f"Saving best model and tokenizer to {MODEL_SAVE_PATH}...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

logging.info(f"Process complete! Your {NUM_SHOTS}-shot model is ready.")
