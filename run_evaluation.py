# run_evaluation.py (最终对比评测版)

import torch
import pandas as pd
import logging
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 核心配置：指定要评估的那个“少样本模型”
NUM_SHOTS = 50
MODEL_NAME_OR_PATH = f"./saved_models/trec06c_classifier_{NUM_SHOTS}_shot"

# 测试集路径
CLEAN_TEST_PATH = "./data/test.csv"
CONFUSED_TEST_PATH = "./data/test_confused.csv"

# 最终对比报告的保存路径
REPORT_SAVE_PATH = f"./evaluation_report_comparison_{NUM_SHOTS}_shot.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_on_dataset(model, tokenizer, file_path, description="Evaluating"):
    """
    在一个指定的数据集上评测模型性能的通用函数。
    返回一个包含各项指标的字典。
    """
    logging.info(f"Starting evaluation on dataset: {file_path}")

    # 1. 加载并准备数据
    try:
        df_test = pd.read_csv(file_path).dropna()
        texts = df_test["text"].astype(str).tolist()
        true_labels = df_test["label"].tolist()
    except FileNotFoundError:
        logging.error(f"Dataset not found at {file_path}. Please generate it first.")
        return None

    logging.info("Tokenizing data...")
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    # 2. 批量预测
    all_preds = []
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            batch_input_ids, batch_attention_mask = [b.to(DEVICE) for b in batch]
            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

    # 3. 计算指标
    tn, fp, fn, tp = confusion_matrix(true_labels, all_preds).ravel()
    report = {
        "dataset_name": os.path.basename(file_path),
        "total_samples": len(true_labels),
        "accuracy": f"{accuracy_score(true_labels, all_preds):.4f}",
        "precision": f"{precision_score(true_labels, all_preds, zero_division=0):.4f}",
        "recall": f"{recall_score(true_labels, all_preds, zero_division=0):.4f}",
        "f1_score": f"{f1_score(true_labels, all_preds, zero_division=0):.4f}",
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
    }
    return report


def main():
    logging.info(f"Starting comparative evaluation for model: {MODEL_NAME_OR_PATH}...")

    # 1. 加载我们训练好的模型和分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
        model.to(DEVICE)
        model.eval()
    except OSError:
        logging.error(
            f"Model not found at {MODEL_NAME_OR_PATH}. Please run train_trec06c.py first."
        )
        return

    # 2. 在“干净测试集”上进行评测
    clean_report = evaluate_on_dataset(
        model, tokenizer, CLEAN_TEST_PATH, "Evaluating on Clean Set"
    )

    # 3. 在“混淆测试集”上进行评测
    confused_report = evaluate_on_dataset(
        model, tokenizer, CONFUSED_TEST_PATH, "Evaluating on Confused Set"
    )

    if not clean_report or not confused_report:
        logging.error("Evaluation could not be completed for one or both datasets.")
        return

    # 4. 整合报告
    final_comparison_report = {
        "model_name": f"{NUM_SHOTS}-shot Fine-Tuned Model",
        "model_path": MODEL_NAME_OR_PATH,
        "clean_set_report": clean_report,
        "confused_set_report": confused_report,
    }

    # 5. 保存并打印最终的对比报告
    logging.info(f"Saving final comparison report to {REPORT_SAVE_PATH}...")
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_comparison_report, f, ensure_ascii=False, indent=4)

    print("\n========================================================")
    print("           FINAL COMPARISON REPORT")
    print("========================================================")
    print(json.dumps(final_comparison_report, indent=4, ensure_ascii=False))
    print("========================================================")


if __name__ == "__main__":
    main()
