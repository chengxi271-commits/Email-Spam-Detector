<<<<<<< HEAD
中文垃圾邮件分类器鲁棒性分析
本项目旨在探索预训练语言模型（hfl/bert-wwm）在中文垃圾邮件分类任务中的性能，并着重于评估其在面对多种文本扰动攻击时的鲁棒性。

整个实验流程分为两个核心阶段：

基线建立：通过少样本微调 (Few-Shot Fine-Tuning) 策略，建立一个性能适中、具备明确可衡量改进空间的基线模型。

鲁棒性测试：创建一个包含多种混淆策略的“脏”测试集，评估基线模型在上面的性能衰减情况。

核心功能
科学的基线建立方法：采用少样本微调，精准控制基线模型的性能，为后续优化研究提供稳定的参照点。

高级混淆数据生成：内置独立的混淆数据生成脚本 (create_confused_set.py)，融合了形近字、繁体字、同音字、拼音替换和汉字部件拆分五大策略。

一键化对比评估：提供统一的评估脚本 (run_evaluation.py)，可自动在“干净”和“混淆”两个测试集上运行评测，并生成一份结构化的JSON对比报告。

技术栈
核心: Python, PyTorch, Hugging Face Transformers

数据处理: Pandas, Scikit-learn

混淆工具: word.json 字典, shape_similar.txt 知识库, cjklib (用于汉字拆分)

如何复现实验

1. 环境设置

# 创建并激活Conda环境

conda create --name bert_exp python=3.10 -y
conda activate bert_exp

# 安装所有依赖

pip install -r requirements.txt
pip install cjklib # cjklib需要单独安装

2. 数据与模型准备
预训练模型: 将 hfl/chinese-bert-wwm-ext 手动下载并放置于 ./local_models/hfl-bert-wwm/ 目录下。

数据集: 将 trec06c 原始数据集解压，确保其 data 和 full/index 位于 ./data/trec06c/ 路径下。

混淆知识库: 确保 ./data/confusion_sets/ 目录下包含 word.json, shape_similar.txt, 和 char_decomposition.txt。

3. 完整实验流程

# 第一步 (仅需一次): 运行数据预处理，生成train/validation/test三个CSV文件

python preprocess_trec06c.py

# 第二步: 训练少样本基线模型 (可在脚本内调整NUM_SHOTS)

python train_trec06c.py

# 第三步 (可选，仅需一次): 生成用于压力测试的混淆数据集

python create_confused_set.py

# 第四步: 运行对比评估，检验模型在干净和混淆数据集上的表现

# (可在脚本内调整NUM_SHOTS以评估对应的模型)

python run_evaluation.py
=======
# 基于BERT的中文垃圾邮件检测与模型对比平台

本项目是一个端到端的深度学习应用，旨在利用预训练语言模型（BERT）对中文邮件进行垃圾信息识别。项目实现了从数据处理、模型微调、离线评测到Web应用部署的全流程，并搭建了一个可交互的前端界面，用于实时展示和对比不同模型的性能。

## 核心功能

- **科学的数据处理流程**：内置独立的脚本，可将原始`trec06c`邮件语料进行清洗，并严格划分为训练、验证和测试集。
- **双模型对比框架**：
    1.  **基线模型**：基于`hfl/chinese-bert-wwm-ext`进行标准微调。
    2.  **增强模型**：基于`ChineseBERT`论文思想，在输入端融合了简化的字形和拼音特征。
- **一键批量评测**：提供独立的评测脚本`run_evaluation.py`，可在整个独立测试集上对两个模型进行全面的性能评估，生成包括准确率、精确率、召回率、F1分数和混淆矩阵在内的专业报告。
- **可交互Web应用**：
    - 使用Flask和Gunicorn部署，提供API服务。
    - 前端界面支持对两个模型进行实时的单条文本预测对比。
    - 支持一键加载并展示完整的离线评测报告。

## 技术栈

- **后端**: Python, PyTorch, Hugging Face Transformers, Flask, Gunicorn
- **前端**: HTML, CSS, JavaScript (原生)
- **数据处理**: Pandas, Scikit-learn

## 如何运行

1.  **环境设置**:
    ```bash
    # 创建并激活Conda环境
    conda create --name bert_exp python=3.10 -y
    conda activate bert_exp

    # 安装所有依赖
    pip install torch torchvision torchaudio transformers scikit-learn pandas flask gunicorn pypinyin tqdm
    ```

2.  **数据与模型准备**:
    * 将预训练模型 `hfl/chinese-bert-wwm-ext` 手动下载并放置于 `local_models` 目录下。
    * 将 `trec06c.zip` 数据集解压并放置于 `data` 目录下。

3.  **执行流程**:
    ```bash
    # 第一步 (只做一次): 运行数据预处理，生成train/validation/test三个CSV文件
    python preprocess_trec06c.py

    # 第二步: 训练两个模型 (可以并行或依次执行)
    python train_trec06c.py # 训练标准BERT
    python train_chinese_bert.py # 训练增强版ChineseBERT

    # 第三步 (可选): 运行离线批量评测，生成JSON报告
    python run_evaluation.py

    # 第四步: 启动Web服务
    gunicorn -w 2 -b 0.0.0.0:8888 app:app
    ```
>>>>>>> ecfde0c... Create README.md
