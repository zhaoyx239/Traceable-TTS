Audio Classification: Real vs Synthetic
本项目利用 Wav2Vec2 模型实现了一个简单的音频分类系统，用于区分合成音频和真人音频。该系统可用于检测音频是否为合成生成的音频。

目录结构
php
复制代码
audio_classification_project/
├── data/
│   ├── real/                  # 存放真人音频数据
│   ├── synthetic/             # 存放合成音频数据
├── models/
│   └── wav2vec_classifier.py  # 定义基于 Wav2Vec2 的分类模型
├── utils/
│   ├── preprocess.py          # 音频加载与预处理
├── train.py                   # 训练模型的主脚本
├── test.py                    # 测试模型的脚本
├── infer.py                   # 推理脚本
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
安装
克隆本仓库：

bash
复制代码
git clone <repository_url>
cd audio_classification_project
安装项目依赖：

bash
复制代码
pip install -r requirements.txt
数据准备
将真人音频文件放置在 data/real/ 文件夹中，将合成音频文件放置在 data/synthetic/ 文件夹中。确保音频文件的采样率为 16kHz，或在数据加载时通过预处理函数进行调整。

使用说明
1. 训练模型
运行 train.py 进行模型训练，数据将会自动划分为训练集、验证集和测试集。

bash
复制代码
python train.py
训练过程中会输出每个 epoch 的训练损失和验证损失。训练完成后，模型权重将保存在 wav2vec_audio_classifier.pth 文件中。

2. 测试模型
运行 test.py 评估模型在测试集上的性能。

bash
复制代码
python test.py
脚本会输出模型在测试集上的分类准确率。

3. 推理
使用 infer.py 对单个音频文件进行分类推理，判断其是真人音频还是合成音频。

bash
复制代码
python infer.py
在 infer.py 中修改音频文件路径以运行推理。

项目文件说明
data/：存放音频数据文件夹。
real/：真人音频文件。
synthetic/：合成音频文件。
models/：存放模型定义文件。
wav2vec_classifier.py：基于 Wav2Vec2 的分类器模型定义。
utils/：存放预处理代码。
preprocess.py：定义音频加载和预处理函数。
train.py：训练模型的主脚本，包含数据加载、模型训练和验证逻辑。
test.py：测试模型在测试集上的性能。
infer.py：对输入音频文件进行推理，判断其是真人音频还是合成音频。
requirements.txt：列出项目所需的依赖项。
README.md：项目说明文档。
参考
本项目使用了 Hugging Face Transformers 库中的 Wav2Vec2 预训练模型