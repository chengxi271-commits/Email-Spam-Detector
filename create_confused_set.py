# create_confused_set.py (最终修正版 - 无外部库依赖)

import pandas as pd
import json
import random
from tqdm import tqdm

# --- 配置 ---
TOTAL_CONFUSION_PROB = 0.05
SOURCE_TEST_SET_PATH = "./data/test.csv"
CONFUSED_TEST_SET_PATH = "./data/test_confused.csv"
PINYIN_TRAD_DICT_PATH = "./data/confusion_sets/word.json"
SHAPE_SIMILAR_DICT_PATH = "./data/confusion_sets/shape_similar.txt"
# 新增：我们最终的汉字拆分字典路径
DECOMPOSITION_DICT_PATH = "./data/confusion_sets/char_decomposition.txt"


class Confuser:
    def __init__(self, pinyin_trad_path, shape_path, decomposition_path):
        print("Initializing Confuser with local dictionary files...")

        self.pinyin_map = {}
        self.char_to_pinyin = {}
        self.traditional_map = {}
        self.shape_similar_map = {}
        self.decomposition_map = {}  # 新增：用于存放汉字部件

        self.load_pinyin_traditional_dict(pinyin_trad_path)
        self.load_shape_similar_dict(shape_path)
        self.load_decomposition_dict(decomposition_path)

        self.strategies = [
            (self.replace_with_homophone, 3),
            (self.replace_with_shape_similar, 3),
            (self.replace_with_traditional, 2),
            (self.replace_with_pinyin, 1),
            (self.split_char, 2),
        ]
        self.strategy_functions, self.strategy_weights = zip(*self.strategies)
        print("Confuser initialized successfully.")

    def load_pinyin_traditional_dict(self, path):
        # 此函数保持不变
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in tqdm(data, desc="Processing Pinyin/Traditional Dictionary"):
            char, pinyin, old_word = (
                item["word"],
                item.get("pinyin", ""),
                item.get("oldword", ""),
            )
            pinyin = pinyin.split(",")[0].strip()
            if pinyin:
                self.char_to_pinyin[char] = pinyin
                self.pinyin_map.setdefault(pinyin, []).append(char)
            if old_word and old_word != char:
                self.traditional_map[char] = old_word

    def load_shape_similar_dict(self, path):
        # 此函数保持不变
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing Shape-Similar Dictionary"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.shape_similar_map[parts[0]] = parts[1:]

    def load_decomposition_dict(self, path):
        """新增：加载汉字部件分解字典"""
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing Decomposition Dictionary"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    # 文件格式: 目标字 = 部件1 + 部件2
                    target_char, components = parts[0], parts[2:]
                    self.decomposition_map[target_char] = components

    # --- 五大混淆策略 ---
    def replace_with_homophone(self, char):
        pinyin = self.char_to_pinyin.get(char)
        if pinyin and len(self.pinyin_map.get(pinyin, [])) > 1:
            candidates = [c for c in self.pinyin_map[pinyin] if c != char]
            if candidates:
                return random.choice(candidates)
        return char

    def replace_with_traditional(self, char):
        return self.traditional_map.get(char, char)

    def replace_with_shape_similar(self, char):
        candidates = self.shape_similar_map.get(char)
        if candidates:
            return random.choice(candidates)
        return char

    def replace_with_pinyin(self, char):
        pinyin = self.char_to_pinyin.get(char)
        if pinyin:
            return f" {''.join(filter(str.isalpha, pinyin))} "
        return char

    def split_char(self, char):
        """策略五：字符拆分 (使用字典)"""
        parts = self.decomposition_map.get(char)
        if parts:
            return " ".join(parts)
        return char

    def confuse_sentence(self, sentence):
        new_sentence = ""
        for char in sentence:
            if "\u4e00" <= char <= "\u9fff" and random.random() < TOTAL_CONFUSION_PROB:
                strategy = random.choices(
                    self.strategy_functions, weights=self.strategy_weights, k=1
                )[0]
                new_sentence += strategy(char)
            else:
                new_sentence += char
        return new_sentence


def main():
    print("Starting confusion set generation process...")
    confuser = Confuser(
        PINYIN_TRAD_DICT_PATH, SHAPE_SIMILAR_DICT_PATH, DECOMPOSITION_DICT_PATH
    )

    print(f"Reading source test set from {SOURCE_TEST_SET_PATH}...")
    df = pd.read_csv(SOURCE_TEST_SET_PATH).dropna().astype(str)

    tqdm.pandas(desc="Applying confusion to sentences")
    df["text"] = df["text"].progress_apply(confuser.confuse_sentence)

    print(f"Saving confused test set to {CONFUSED_TEST_SET_PATH}...")
    df.to_csv(CONFUSED_TEST_SET_PATH, index=False, encoding="utf-8")

    print("\nConfusion set generation complete!")
    print(f"A new confused test set has been saved at: {CONFUSED_TEST_SET_PATH}")


if __name__ == "__main__":
    main()
