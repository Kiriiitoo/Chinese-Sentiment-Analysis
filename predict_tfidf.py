# predict_tfidf.py
import joblib
import jieba
import numpy as np

# ---- 定义与训练时一致的分词函数（必须在加载前） ----
def my_tokenizer(x):
    return x.split()

# ---- 加载模型与向量器 ----
vect = joblib.load("model/tfidf_vectorizer.joblib")
clf = joblib.load("model/logistic.joblib")

# ---- 中文分词函数 ----
def preprocess_text(text):
    return " ".join(jieba.lcut(text))

print("✅ 中文情感分析模型已加载，可以开始预测（输入 q 退出）")

# ---- 交互循环 ----
while True:
    text = input("\n请输入一句话：")
    if text.lower() == "q":
        print("程序已退出。")
        break

    # 文本分词
    cut_text = preprocess_text(text)

    # 向量化
    X = vect.transform([cut_text])

    # 预测结果与概率
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]  # [负面概率, 正面概率]

    sentiment = "正面" if pred == 1 else "负面"
    confidence = prob[pred]  # 取预测类别的置信度

    print(f"预测结果：{sentiment}（置信度：{confidence:.2f}）")
