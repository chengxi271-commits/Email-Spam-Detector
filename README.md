# 中文垃圾邮件分类器鲁棒性分析

本项目旨在探索预训练语言模型（`hfl/bert-wwm`）在中文垃圾邮件分类任务中的性能，并着重于评估其在面对多种文本扰动攻击时的**鲁棒性**。

整个实验流程分为两个核心阶段：

1.  **基线建立**：通过**少样本微调 (Few-Shot Fine-Tuning)** 策略，建立一个性能适中、具备明确可衡量改进空间的基线模型。
2.  **鲁棒性测试**：创建一个包含多种混淆策略的“脏”测试集，评估基线模型在上面的性能衰减情况。

## 核心功能

-   **科学的基线建立方法**：采用少样本微调，精准控制基线模型的性能，为后续优化研究提供稳定的参照点。
-   **模块化的混淆数据生成**：内置一个独立的、可复用的混淆工具模块 (`utils_confusion.py`)，并提供一个生成脚本 (`create_confused_set.py`)，可融合**形近字、繁体字、同音字、拼音替换和汉字部件拆分**五大策略。
-   **一键化对比评估**：提供统一的评估脚本 (`run_evaluation.py`)，可自动在“干净”和“混淆”两个测试集上运行评测，并生成一份结构化的JSON对比报告。

## 技术栈

-   **核心**: Python, PyTorch, Hugging Face Transformers
-   **数据处理**: Pandas, Scikit-learn
-   **混淆工具**: 完全基于本地数据文件，无额外Python库依赖。
    -   `word.json` (用于同音字、繁简转换)
    -   `shape_similar.txt` (用于形近字)
    -   `char_decomposition.txt` (用于汉字拆分)

---

## 如何复现实验

### 1. 准备工作

-   **预训练模型**: 将 `hfl/chinese-bert-wwm-ext` 手动下载并放置于 `./local_models/hfl-bert-wwm/` 目录下。
-   **数据集**: 将 `trec06c` 原始数据集解压，确保其 `data` 和 `full/index` 位于 `./data/trec06c/` 路径下。
-   **混淆知识库**: 确保 `./data/confusion_sets/` 目录下包含 `word.json`, `shape_similar.txt`, 和 `char_decomposition.txt`。

### 2. 完整执行步骤

```bash
# --- 步骤 1: 环境设置 ---
# 创建并激活Conda环境
conda create --name bert_exp python=3.10 -y
conda activate bert_exp

# 安装所有核心依赖
pip install torch transformers scikit-learn pandas tqdm

# --- 步骤 2: 数据预处理 (仅需一次) ---
# 该命令会生成train/validation/test三个CSV文件
python preprocess_trec06c.py

# --- 步骤 3: 训练少样本基线模型 ---
# 注意：可在 train_trec06c.py 脚本内调整 NUM_SHOTS 变量
python train_trec06c.py

# --- 步骤 4: 生成混淆数据集 (可选) ---
# 该命令会根据 test.csv 生成 test_confused.csv
python create_confused_set.py

# --- 步骤 5: 运行对比评估 ---
# 注意：确保 run_evaluation.py 内的 NUM_SHOTS 与训练时一致
python run_evaluation.py
