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
