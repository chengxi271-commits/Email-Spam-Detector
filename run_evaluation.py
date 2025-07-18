# run_evaluation.py (Final, Single-GPU, Correct Version)

import torch
import pandas as pd
import logging
import json
import os
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from char_processor import enhance_text

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 路径定义 ---
MODEL_A_PATH = './saved_models/trec06c_classifier_v2'
MODEL_B_PATH = './saved_models/chinese_bert_enhanced_v1'
TEST_DATA_PATH = './data/test.csv'
REPORT_SAVE_PATH = './evaluation_report_final.json'

# --- 定义设备 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 计算指标的函数 (和之前一样) ---
def calculate_metrics(true_labels, predicted_labels, model_name):
    logging.info(f"Calculating metrics for {model_name}...")
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    return {
        "model_name": model_name, "total_samples": len(true_labels),
        "accuracy": f"{accuracy:.4f}", "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}", "f1_score": f"{f1:.4f}",
        "confusion_matrix": {
            "true_positives": int(tp), "false_positives": int(fp),
            "true_negatives": int(tn), "false_negatives": int(fn)
        }
    }

# --- 核心评测函数 (单GPU版) ---
def evaluate_model(model, tokenizer, texts):
    model.eval()
    all_predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating on {DEVICE}"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(preds.cpu().numpy())
    return all_predictions

def main():
    logging.info("Starting SINGLE-GPU evaluation...")
    
    # 1. 加载测试数据
    df_test = pd.read_csv(TEST_DATA_PATH).dropna()
    true_labels = df_test['label'].tolist()
    original_texts = df_test['text'].astype(str).tolist()

    # 2. 增强文本
    logging.info("Enhancing test texts for ChineseBERT model...")
    tqdm.pandas(desc="Enhancing test set")
    enhanced_texts = df_test['text'].astype(str).progress_apply(enhance_text).tolist()

    # 3. 评估模型A
    logging.info(f"Loading Model A (Standard BERT)...")
    model_a = BertForSequenceClassification.from_pretrained(MODEL_A_PATH).to(DEVICE)
    tokenizer_a = BertTokenizer.from_pretrained(MODEL_A_PATH)
    predictions_a = evaluate_model(model_a, tokenizer_a, original_texts)
    report_a = calculate_metrics(true_labels, predictions_a, "Standard BERT")

    # 4. 评估模型B
    logging.info(f"Loading Model B (ChineseBERT Enhanced)...")
    model_b = BertForSequenceClassification.from_pretrained(MODEL_B_PATH).to(DEVICE)
    tokenizer_b = BertTokenizer.from_pretrained(MODEL_B_PATH)
    predictions_b = evaluate_model(model_b, tokenizer_b, enhanced_texts)
    report_b = calculate_metrics(true_labels, predictions_b, "ChineseBERT Enhanced")
    
    # 5. 整合并保存报告
    final_report = {"model_a_report": report_a, "model_b_report": report_b}
    logging.info(f"Saving final comparison report to {REPORT_SAVE_PATH}...")
    with open(REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
        
    logging.info("Evaluation complete! Report saved.")
    print("\n--- FINAL COMPARISON REPORT (SINGLE GPU - CORRECTED) ---")
    print(json.dumps(final_report, indent=4, ensure_ascii=False))
    print("---------------------------------------------------------")

if __name__ == '__main__':
    main()