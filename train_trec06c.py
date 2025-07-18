# train_trec06c.py

import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import Dataset
import logging

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义数据集类
class TrecDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- 定义常量 ---
TRAIN_FILE_PATH = './data/train.csv'
VALIDATION_FILE_PATH = './data/validation.csv'
MODEL_SAVE_PATH = './saved_models/trec06c_classifier_v2' # 新模型的新名字
MODEL_NAME = './local_models/hfl-bert-wwm' # 从本地加载预训练模型

# --- 加载和预处理数据 ---
logging.info(f"Loading train data from {TRAIN_FILE_PATH}...")
try:
    df_train = pd.read_csv(TRAIN_FILE_PATH).dropna()
    logging.info(f"Train data loaded with {len(df_train)} samples.")
except FileNotFoundError:
    logging.error(f"Train data file not found at {TRAIN_FILE_PATH}. Please run preprocess_trec06c.py first.")
    exit()

logging.info(f"Loading validation data from {VALIDATION_FILE_PATH}...")
try:
    df_val = pd.read_csv(VALIDATION_FILE_PATH).dropna()
    logging.info(f"Validation data loaded with {len(df_val)} samples.")
except FileNotFoundError:
    logging.error(f"Validation data file not found at {VALIDATION_FILE_PATH}. Please run preprocess_trec06c.py first.")
    exit()

# 分别准备训练和验证的文本与标签
train_texts, train_labels = df_train['text'].astype(str).tolist(), df_train['label'].tolist()
val_texts, val_labels = df_val['text'].astype(str).tolist(), df_val['label'].tolist()


# --- 初始化分词器并处理文本 ---
logging.info(f"Loading tokenizer from local path: {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

logging.info("Tokenizing datasets...")
# 邮件内容较长，我们将最大长度设为512
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

train_dataset = TrecDataset(train_encodings, train_labels)
val_dataset = TrecDataset(val_encodings, val_labels)


# --- 加载模型 ---
logging.info(f"Loading pre-trained model from local path: {MODEL_NAME}...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# --- 定义训练参数 (使用我们最终验证过的兼容版) ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=12, # 这是单GPU的设置，如果用4卡，等效batch size是32
    per_device_eval_batch_size=48,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=200,                # 每200步打印一次日志
    evaluation_strategy="steps",      # 策略改为按步数评估
    eval_steps=1000,                  # 每1000步在验证集上评估一次
    save_strategy="steps",            # 策略改为按步数保存
    save_steps=1000,                  # 每1000步保存一次模型检查点
    load_best_model_at_end=True,      # 训练结束后加载最优模型
    metric_for_best_model="accuracy", # 以准确率作为最优模型标准
    fp16=True, 
)

# --- 定义评估指标 ---
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

logging.info("Starting model fine-tuning on trec06c dataset...")
trainer.train()
logging.info("Training finished.")


# --- 保存最好的模型 ---
logging.info(f"Saving best model and tokenizer to {MODEL_SAVE_PATH}...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

logging.info("Process complete! New model is ready for deployment.")