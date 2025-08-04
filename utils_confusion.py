# utils_confusion.py

import json
import random
from tqdm import tqdm


class Confuser:
    """
    一个集成了多种专业级中文混淆策略的工具类。
    这个类是可复用的，可以被任何需要生成混淆数据的脚本导入和使用。
    """

    def __init__(self, pinyin_trad_path, shape_path, decomposition_path):
        print("Initializing Confuser with local dictionary files...")

        self.pinyin_map = {}
        self.char_to_pinyin = {}
        self.traditional_map = {}
        self.shape_similar_map = {}
        self.decomposition_map = {}

        self.load_pinyin_traditional_dict(pinyin_trad_path)
        self.load_shape_similar_dict(shape_path)
        self.load_decomposition_dict(decomposition_path)

        # 定义所有可用的混淆函数及其权重
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
        """加载并预处理JSON字典，用于同音字和繁体字"""
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
        """加载形近字字典"""
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing Shape-Similar Dictionary"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.shape_similar_map[parts[0]] = parts[1:]

    def load_decomposition_dict(self, path):
        """加载汉字部件分解字典 (已修正)"""
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing Decomposition Dictionary"):
                # ####################################################################
                # 核心修正点：
                # 1. 先用制表符 '\t' 分割，将原字和不同的拆分方法分开。
                # 2. 只取第一个拆分方法 (parts[1])。
                # 3. 再用空格分割，得到最终的部件列表。
                # ####################################################################
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    target_char = parts[0]
                    # 只取第一个拆分方法
                    first_decomposition = parts[1]
                    # 将第一个拆分方法按空格分割成部件
                    components = first_decomposition.split()
                    if components:  # 确保部件列表不为空
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
        parts = self.decomposition_map.get(char)
        if parts:
            return " ".join(parts)
        return char

    def confuse_sentence(self, sentence, prob):
        """
        对整个句子以指定的概率应用多种混淆策略
        :param sentence: 输入的句子
        :param prob: 每个字符被混淆的概率
        :return: 混淆后的句子
        """
        new_sentence = ""
        for char in sentence:
            if "\u4e00" <= char <= "\u9fff" and random.random() < prob:
                strategy = random.choices(
                    self.strategy_functions, weights=self.strategy_weights, k=1
                )[0]
                new_sentence += strategy(char)
            else:
                new_sentence += char
        return new_sentence
