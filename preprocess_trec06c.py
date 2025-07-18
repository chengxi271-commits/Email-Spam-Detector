# preprocess_trec06c.py (Final Offline Processing Script)

import os
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from char_processor import enhance_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 路径定义 ---
BASE_PATH = './data/trec06c'
INDEX_FILE_PATH = os.path.join(BASE_PATH, 'full/index')
TOKENIZER_PATH = './local_models/hfl-bert-wwm'
OUTPUT_DIR = './data/'

def parse_email(file_path):
    try:
        with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
            first_blank_line_index = -1
            for i, line in enumerate(lines):
                if line.strip() == '':
                    first_blank_line_index = i
                    break
            body = ''.join(lines[first_blank_line_index+1:]) if first_blank_line_index != -1 else ''.join(lines)
            return body.strip()
    except Exception:
        return None

def main():
    # 1. 加载原始数据
    all_data = []
    with open(INDEX_FILE_PATH, 'r', encoding='utf-8') as f:
        index_lines = f.readlines()
    for line in tqdm(index_lines, desc="Reading raw emails"):
        parts = line.strip().split()
        if len(parts) == 2:
            label_str, relative_path = parts
            label = 1 if label_str == 'spam' else 0
            email_file_path = os.path.abspath(os.path.join(os.path.dirname(INDEX_FILE_PATH), relative_path))
            text = parse_email(email_file_path)
            if text:
                all_data.append([text, label])
    df_full = pd.DataFrame(all_data, columns=['text', 'label'])
    logging.info(f"Successfully processed {len(df_full)} raw emails.")

    # 2. 划分数据集
    df_train, df_temp = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['label'])
    df_validation, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])

    datasets = {'train': df_train, 'validation': df_validation, 'test': df_test}
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

    # 3. 对每一个数据集进行增强、分词并保存
    for name, df in datasets.items():
        logging.info(f"Processing '{name}' dataset with {len(df)} samples...")
        
        # 增强
        tqdm.pandas(desc=f"Enhancing {name} set")
        enhanced_texts = df['text'].astype(str).progress_apply(enhance_text)
        
        # 分词
        logging.info(f"Tokenizing '{name}' set...")
        encodings = tokenizer(enhanced_texts.tolist(), truncation=True, padding=True, max_length=512)
        
        # 保存为可以直接加载的PyTorch文件
        data_dict = {'encodings': encodings, 'labels': df['label'].tolist()}
        save_path = os.path.join(OUTPUT_DIR, f'{name}_enhanced_dataset.pt')
        torch.save(data_dict, save_path)
        logging.info(f"Saved preprocessed '{name}' dataset to {save_path}")

    logging.info("Offline preprocessing for ChineseBERT complete!")

if __name__ == '__main__':
    main()