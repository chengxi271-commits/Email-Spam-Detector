# char_processor.py

from pypinyin import pinyin, Style

# 为了简化，我们先硬编码一个小的字形信息字典
# 它的原理是模拟通过图像分析或查表得出的汉字结构特征
# 's'代表简单结构, 'c'代表复杂结构, 'lr'代表左右结构, 'ud'代表上下结构
# 在真实的研究中，这里会是一个更复杂的特征向量
GLYPH_FEATURE_DICT = {
    '你': 'lr', '好': 'lr', '我': 's', '他': 'lr', '是': 'ud', '的': 'lr',
    '发': 'ud', '票': 'ud', '代': 'lr', '开': 's', '保': 'lr', '真': 'ud',
    '恭': 'ud', '喜': 'ud', '中': 's', '奖': 'ud', '了': 's', '请': 'lr',
    '点': 'ud', '击': 'lr', '链': 'lr', '接': 'lr', '领': 'lr', '取': 'lr',
    '晚': 'lr', '上': 's', '一': 's', '起': 'lr', '吃': 'lr', '饭': 'lr', '吗': 'lr',
}

def get_glyph_feature(char):
    """获取单个汉字的简化版字形特征"""
    return GLYPH_FEATURE_DICT.get(char, 'u') # 'u' for unknown

def enhance_text(sentence: str) -> str:
    """
    接收一个句子，返回带有拼音和字形特征的增强版句子。
    这是一个简化的实现，用于验证思路。
    """
    if not isinstance(sentence, str):
        return ""

    enhanced_parts = []
    # lazy=False 可以处理多音字，但为了简化，我们先用默认的
    # pinyin函数的返回值是一个二维列表，例如 [['nǐ'], ['hǎo']]
    pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=False)

    for i, char in enumerate(sentence):
        # 只处理中文字符
        if '\u4e00' <= char <= '\u9fff':
            # 获取拼音, pinyin_list[i]是['nǐ']，pinyin_list[i][0]是'ni3'
            pinyin_str = pinyin_list[i][0] if i < len(pinyin_list) else ''
            
            # 获取字形特征
            glyph_str = get_glyph_feature(char)
            
            # 拼接成 "原字 [拼音:xx] [字形:yy]" 的格式
            enhanced_parts.append(f"{char} [pinyin:{pinyin_str}] [glyph:{glyph_str}]")
        else:
            enhanced_parts.append(char)
            
    return ' '.join(enhanced_parts)


# --- 可以在这里测试一下效果 ---
if __name__ == '__main__':
    test_sentence_1 = "你好，发票代开"
    enhanced_1 = enhance_text(test_sentence_1)
    print(f"Original: {test_sentence_1}")
    print(f"Enhanced: {enhanced_1}")
    # 期望输出: 你 [pinyin:ni3] [glyph:lr] 好 [pinyin:hao3] [glyph:lr] , 发 [pinyin:fa1] [glyph:ud] 票 [pinyin:piao4] [glyph:ud] 代 [pinyin:dai4] [glyph:lr] 开 [pinyin:kai1] [glyph:s]

    test_sentence_2 = "晚上吃饭吗？"
    enhanced_2 = enhance_text(test_sentence_2)
    print(f"\nOriginal: {test_sentence_2}")
    print(f"Enhanced: {enhanced_2}")
    # 期望输出: 晚 [pinyin:wan3] [glyph:lr] 上 [pinyin:shang4] [glyph:s] 吃 [pinyin:chi1] [glyph:lr] 饭 [pinyin:fan4] [glyph:lr] 吗 [pinyin:ma5] [glyph:lr] ?