# create_confused_set.py

import pandas as pd
from tqdm import tqdm
from utils_confusion import Confuser  # <-- 从我们新建的模块中导入核心类

# --- 配置 ---
TOTAL_CONFUSION_PROB = 0.05  # 每个中文字符被混淆的总概率

# 输入和输出文件路径
SOURCE_TEST_SET_PATH = './data/test.csv'
CONFUSED_TEST_SET_PATH = './data/test_confused.csv'
PINYIN_TRAD_DICT_PATH = './data/confusion_sets/word.json'
SHAPE_SIMILAR_DICT_PATH = './data/confusion_sets/shape_similar.txt'
DECOMPOSITION_DICT_PATH = './data/confusion_sets/char_decomposition.txt'

def main():
    """
    主函数，负责调用混淆工具来生成混淆数据集。
    """
    print("Starting confusion set generation process...")
    
    # 1. 初始化混淆器
    confuser = Confuser(PINYIN_TRAD_DICT_PATH, SHAPE_SIMILAR_DICT_PATH, DECOMPOSITION_DICT_PATH)
    
    # 2. 读取源测试集
    print(f"Reading source test set from {SOURCE_TEST_SET_PATH}...")
    df = pd.read_csv(SOURCE_TEST_SET_PATH).dropna().astype(str)
    
    # 3. 对'text'列应用混淆函数
    # 使用 lambda 函数来传递额外的概率参数
    tqdm.pandas(desc="Applying confusion to sentences")
    df['text'] = df['text'].progress_apply(
        lambda s: confuser.confuse_sentence(s, prob=TOTAL_CONFUSION_PROB)
    )
    
    # 4. 保存混淆后的数据集
    print(f"Saving confused test set to {CONFUSED_TEST_SET_PATH}...")
    df.to_csv(CONFUSED_TEST_SET_PATH, index=False, encoding='utf-8')
    
    print("\nConfusion set generation complete!")
    print(f"A new confused test set has been saved at: {CONFUSED_TEST_SET_PATH}")

if __name__ == '__main__':
    main()
