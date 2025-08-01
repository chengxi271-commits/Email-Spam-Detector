# preprocess_trec06c.py (路径拼接已修正的最终版)

import os
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 路径定义 ---
BASE_PATH = "./data/trec06c"
INDEX_FILE_PATH = os.path.join(BASE_PATH, "full/index")
OUTPUT_DIR = "./data/"


def parse_email(file_path):
    """解析单个邮件文件，提取其正文内容。"""
    try:
        with open(file_path, "r", encoding="gb18030", errors="ignore") as f:
            lines = f.readlines()
            first_blank_line_index = -1
            for i, line in enumerate(lines):
                if line.strip() == "":
                    first_blank_line_index = i
                    break
            body = (
                "".join(lines[first_blank_line_index + 1 :])
                if first_blank_line_index != -1
                else "".join(lines)
            )
            return body.strip()
    except Exception as e:
        # 在这里我们不打印警告，因为如果路径错误，会产生海量无用信息
        return None


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging.info("开始数据预处理...")

    # 1. 加载原始数据索引
    all_data = []
    try:
        with open(INDEX_FILE_PATH, "r", encoding="utf-8") as f:
            index_lines = f.readlines()
    except FileNotFoundError:
        logging.error(
            f"索引文件未在 {INDEX_FILE_PATH} 找到。请确保 trec06c 数据集已正确放置。"
        )
        return

    for line in tqdm(index_lines, desc="正在读取原始邮件"):
        parts = line.strip().split()
        if len(parts) == 2:
            label_str, relative_path = parts
            label = 1 if label_str == "spam" else 0

            # ####################################################################
            # 关键修正点：使用正确的、鲁棒的路径拼接方法
            # ####################################################################
            email_file_path = os.path.abspath(
                os.path.join(os.path.dirname(INDEX_FILE_PATH), relative_path)
            )
            # ####################################################################

            text = parse_email(email_file_path)
            if text:
                all_data.append([text, label])

    if not all_data:
        logging.error(
            "未能成功解析任何邮件文件！请检查 'trec06c/full/index' 文件中的路径是否正确，以及对应的邮件文件是否存在。"
        )
        return

    df_full = pd.DataFrame(all_data, columns=["text", "label"])
    logging.info(f"成功处理了 {len(df_full)} 封原始邮件。")

    # 2. 划分数据集
    df_train, df_temp = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full["label"]
    )
    df_validation, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp["label"]
    )

    # 3. 将数据集保存为 CSV 文件
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    validation_path = os.path.join(OUTPUT_DIR, "validation.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8")
    df_validation.to_csv(validation_path, index=False, encoding="utf-8")
    df_test.to_csv(test_path, index=False, encoding="utf-8")

    logging.info(f"数据集保存成功:")
    logging.info(f"- 训练集: {train_path} ({len(df_train)} 条样本)")
    logging.info(f"- 验证集: {validation_path} ({len(df_validation)} 条样本)")
    logging.info(f"- 测试集: {test_path} ({len(df_test)} 条样本)")
    logging.info("预处理完成！")


if __name__ == "__main__":
    main()
