# 中文情感分析系统 (Chinese Sentiment Analysis)

一个面向中文评论的情感分析项目，包含：  
- 基线模型：TF-IDF + Logistic Regression  
- 高精度模型：BERT 微调（`bert-base-chinese`）  
- Web Demo：Streamlit 单句/批量分析、关键词云、性能对比

---

## 仓库结构（重要）
Chinese Sentiment Analysis/
├── app.py
├── baseline_tfidf.py
├── bert_finetune.py
├── preprocess.py
├── predict_tfidf.py
├── requirements.txt
├── .gitignore
├── README.md
└── model/ # 不包含大模型（见下文）

---

## 快速开始（本地部署）

1. 克隆仓库（已完成）：
git clone https://github.com/Kiriiitoo/Chinese-Sentiment-Analysis.git
cd Chinese-Sentiment-Analysis
2.建议创建虚拟环境并激活：
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3.安装依赖：
pip install -r requirements.txt
如果没有 requirements.txt，可使用：
pip install joblib jieba scikit-learn pandas numpy matplotlib streamlit torch transformers datasets wordcloud
4.准备模型（必须手动放置 — 见下节“模型下载说明”）。
5.运行 Web Demo：
streamlit run app.py
浏览器打开 http://localhost:8501。

##模型下载说明（重要）
为了避免在仓库托管大文件，本项目不包含预训练或微调后的 BERT 二进制模型与较大的 .joblib 文件。
请按需准备：
TF-IDF / LR（可选）：如果你想直接使用训练好的 tfidf_vectorizer.joblib 与 logistic.joblib，请把它们放到仓库的 model/ 目录下：
model/tfidf_vectorizer.joblib
model/logistic.joblib
BERT 微调模型（可选）：将 model/bert_sentiment/ 文件夹放入仓库 model/ 目录下（包含 pytorch_model.bin、config.json、vocab.txt 等）。
模型下载链接:通过网盘分享的文件：model.zip
链接: https://pan.baidu.com/s/1OFOcY4f84FOk0ifiXzTSQg?pwd=yt1a 提取码: yt1a 
