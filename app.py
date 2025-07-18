# app.py (Final version with corrected report path)

from flask import Flask, request, jsonify, render_template
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os
import json
import logging

# --- 1. 初始化应用和路径 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_project_root = os.path.dirname(os.path.realpath(__file__))
_static_folder = os.path.join(_project_root, 'static')
_template_folder = os.path.join(_project_root, 'templates')

app = Flask(__name__, 
            static_folder=_static_folder, 
            template_folder=_template_folder)


# --- 2. 加载两个需要对比的模型 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型A：标准BERT
MODEL_A_PATH = os.path.join(_project_root, 'saved_models/trec06c_classifier_v2')
model_a, tokenizer_a = None, None
try:
    if os.path.exists(MODEL_A_PATH):
        model_a = BertForSequenceClassification.from_pretrained(MODEL_A_PATH).to(DEVICE)
        model_a.eval()
        tokenizer_a = BertTokenizer.from_pretrained(MODEL_A_PATH)
        logging.info("Model A (Standard BERT) loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Model A: {e}")

# 模型B：ChineseBERT 增强版
MODEL_B_PATH = os.path.join(_project_root, 'saved_models/chinese_bert_enhanced_v1')
model_b, tokenizer_b = None, None
try:
    if os.path.exists(MODEL_B_PATH):
        model_b = BertForSequenceClassification.from_pretrained(MODEL_B_PATH).to(DEVICE)
        model_b.eval()
        tokenizer_b = BertTokenizer.from_pretrained(MODEL_B_PATH)
        logging.info("Model B (ChineseBERT Enhanced) loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Model B: {e}")


# --- 3. 核心预测逻辑 (用于单条实时对比) ---
from char_processor import enhance_text
def get_prediction(model, tokenizer, text, is_enhanced=False):
    if not model or not tokenizer: return {"error": "Model not available."}
    processed_text = enhance_text(text) if is_enhanced else text
    inputs = tokenizer(str(processed_text), return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    return {
        'is_spam': bool(prediction),
        'confidence': {
            'ham': f"{probabilities[0][0].item():.4f}",
            'spam': f"{probabilities[0][1].item():.4f}"
        }
    }


# --- 4. API 端点 ---
@app.route('/predict_comparison', methods=['POST'])
def handle_predict_comparison():
    data = request.get_json()
    if not data or 'text' not in data: return jsonify({'error': 'Missing "text" field'}), 400
    original_text = data['text']
    result_a = get_prediction(model_a, tokenizer_a, original_text, is_enhanced=False)
    result_b = get_prediction(model_b, tokenizer_b, original_text, is_enhanced=True)
    return jsonify({"original_text": original_text, "model_a_result": result_a, "model_b_result": result_b})

# (关键修正) 全面评估接口读取正确的文件名
@app.route('/get_evaluation_report', methods=['GET'])
def get_evaluation_report():
    report_path = os.path.join(_project_root, 'evaluation_report_final.json') # <-- 已修正为_final
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return jsonify(report_data)
    except FileNotFoundError:
        return jsonify({"error": "Evaluation report file not found. Please run 'run_evaluation.py' first."}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read report file: {e}"}), 500


# --- 5. 主页路由 ---
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)