# app.py
import os
import io
import streamlit as st
import joblib
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- transformers 用于 BERT ----
import torch
from transformers import BertTokenizer, BertForSequenceClassification


# ---- 设置中文字体 ----
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

# ---- 定义与训练时一致的分词函数 ----
def my_tokenizer(text):
    # TF-IDF 模型中的 tokenizer 必须与训练时一致，否则无法加载
    return text.split()
# =====================================================
#                 缓存加载 TF-IDF 模型
# =====================================================
@st.cache_resource
def load_tfidf_model():
    vect = joblib.load("model/tfidf_vectorizer.joblib")
    clf = joblib.load("model/logistic.joblib")
    return vect, clf


# =====================================================
#                 缓存加载 BERT 模型
# =====================================================
@st.cache_resource
def load_bert_model():
    if not os.path.exists("model/bert_sentiment"):
        return None, None
    tokenizer = BertTokenizer.from_pretrained("model/bert_sentiment")
    model = BertForSequenceClassification.from_pretrained("model/bert_sentiment")
    model.eval()
    return tokenizer, model


# =====================================================
#                 文本预处理
# =====================================================
def preprocess_text(text):
    return " ".join(jieba.lcut(text))


# =====================================================
#                 Streamlit 页面配置
# =====================================================
st.set_page_config(page_title="中文情感分析系统", page_icon="💬", layout="centered")
st.title("💬 中文情感分析系统")
st.markdown("请输入一句话，或上传 CSV 文件进行批量情感分析。")


# =====================================================
#             侧边栏：模型选择
# =====================================================
st.sidebar.header("⚙️ 模型设置")

model_choice = st.sidebar.radio(
    "选择模型：",
    ("TF-IDF + LR（轻量模型）", "BERT 中文模型（高精度）")
)

use_bert = model_choice.startswith("BERT")

# 加载模型（根据选择）
vect, clf = load_tfidf_model()
tokenizer, bert_model = load_bert_model()


# =====================================================
#             页面选项卡
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔹 单句分析", "📂 批量上传分析", "📈 模型性能对比"])


# =====================================================
# 🔹 单句分析
# =====================================================
with tab1:
    user_input = st.text_area("请输入文本：", height=120, placeholder="例如：这家酒店非常干净，服务态度很好。")

    if st.button("分析情感", key="single"):
        if not user_input.strip():
            st.warning("请输入一句话再进行分析～")
        else:
            # ================== TF-IDF 模型 ==================
            if not use_bert:
                cut_text = preprocess_text(user_input)
                X = vect.transform([cut_text])
                pred = clf.predict(X)[0]
                prob = clf.predict_proba(X)[0]

                sentiment = "正面 😄" if pred == 1 else "负面 😠"
                confidence = prob[pred]

                st.subheader(f"分析结果：{sentiment}")
                st.write(f"置信度：**{confidence:.2f}**")

                # ---- 可视化置信度 ----
                labels = ["负面", "正面"]
                fig, ax = plt.subplots()
                ax.bar(labels, prob, color=["red", "green"])
                ax.set_ylim([0, 1])
                ax.set_ylabel("概率")
                ax.set_title("情感置信度分布")
                st.pyplot(fig)

            # ================== BERT 模型 ==================
            else:
                if tokenizer is None:
                    st.error("❌ 未检测到 BERT 模型，请先运行 bert_finetune.py 进行训练。")
                else:
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred].item()

                    sentiment = "正面 😄" if pred == 1 else "负面 😠"

                    st.subheader(f"BERT 分析结果：{sentiment}")
                    st.write(f"置信度：**{confidence:.2f}**")

                    # 可视化
                    labels = ["负面", "正面"]
                    fig, ax = plt.subplots()
                    ax.bar(labels, probs.numpy()[0], color=["red", "green"])
                    ax.set_ylim([0, 1])
                    ax.set_ylabel("概率")
                    ax.set_title("BERT 情感置信度分布")
                    st.pyplot(fig)


# =====================================================
# 📂 批量上传分析
# =====================================================
with tab2:
    st.write("上传一个 CSV 文件（需包含名为 `text` 的列）。")

    uploaded_file = st.file_uploader("选择文件", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("CSV 文件必须包含列名 `text`。")
            else:
                st.success(f"文件读取成功，共 {len(df)} 条记录。")

                # ================== TF-IDF 批量分析 ==================
                if not use_bert:
                    df["text_cut"] = df["text"].astype(str).apply(preprocess_text)
                    X = vect.transform(df["text_cut"])
                    preds = clf.predict(X)
                    probs = clf.predict_proba(X)

                    df["pred_label"] = preds
                    df["pred_sentiment"] = df["pred_label"].map({1: "正面", 0: "负面"})
                    df["confidence"] = [max(p) for p in probs]

                # ================== BERT 批量分析 ==================
                else:
                    if tokenizer is None:
                        st.error("❌ 未检测到 BERT 模型，请先训练。")
                    else:
                        preds_list, conf_list = [], []
                        for text in df["text"].astype(str).tolist():
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                            with torch.no_grad():
                                outputs = bert_model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                                pred = torch.argmax(probs, dim=1).item()

                            preds_list.append(pred)
                            conf_list.append(probs[0][pred].item())

                        df["pred_label"] = preds_list
                        df["pred_sentiment"] = df["pred_label"].map({1: "正面", 0: "负面"})
                        df["confidence"] = conf_list

                # ==== 显示结果 ====
                st.subheader("📊 分析结果预览")
                st.dataframe(df[["text", "pred_sentiment", "confidence"]].head(10))

                # ==== 关键词云（仅正/负区分） ====
                from wordcloud import WordCloud
                st.subheader("☁️ 情感关键词云")

                font_path = "C:\\Windows\\Fonts\\SimHei.ttf"
                if not os.path.exists(font_path):
                    st.warning("⚠️ 未找到 SimHei.ttf 字体，中文可能会乱码。")

                df["text_cut"] = df["text"].astype(str).apply(preprocess_text)

                # 正面词云
                pos_text = " ".join(df[df["pred_label"] == 1]["text_cut"])
                if pos_text.strip():
                    st.markdown("### 😊 正面评论关键词云")
                    wc = WordCloud(font_path=font_path, background_color="white",
                                   width=600, height=400, colormap="Greens").generate(pos_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

                # 负面词云
                neg_text = " ".join(df[df["pred_label"] == 0]["text_cut"])
                if neg_text.strip():
                    st.markdown("### 😠 负面评论关键词云")
                    wc = WordCloud(font_path=font_path, background_color="white",
                                   width=600, height=400, colormap="Reds").generate(neg_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

                # ==== CSV 下载 ====
                csv_buf = io.BytesIO()
                df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                csv_buf.seek(0)
                st.download_button(
                    label="📥 下载分析结果 CSV",
                    data=csv_buf,
                    file_name="sentiment_result.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"文件读取失败：{e}")

import json

with tab3:
    st.header("📈 模型性能对比")

    # 读取性能文件
    try:
        with open("model/performance.json", "r", encoding="utf-8") as f:
            perf = json.load(f)
    except:
        st.error("⚠️ 未找到 model/performance.json，无法显示性能对比。")
        st.stop()

    tfidf = perf["tfidf"]
    bert = perf["bert"]

    # 数值表格
    st.subheader("🔍 模型指标对比表")
    df_perf = pd.DataFrame({
        "指标": ["准确率 (Accuracy)", "精确率 (Precision)", "召回率 (Recall)", "F1-score"],
        "TF-IDF + LR": [tfidf["accuracy"], tfidf["precision"], tfidf["recall"], tfidf["f1"]],
        "BERT": [bert["accuracy"], bert["precision"], bert["recall"], bert["f1"]]
    })

    st.dataframe(df_perf)

    # 可视化对比（雷达图）
    st.subheader("📊 模型性能雷达图")
    labels = ["准确率", "精确率", "召回率", "F1-score"]
    tfidf_values = [tfidf["accuracy"], tfidf["precision"], tfidf["recall"], tfidf["f1"]]
    bert_values = [bert["accuracy"], bert["precision"], bert["recall"], bert["f1"]]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    tfidf_values += tfidf_values[:1]
    bert_values += bert_values[:1]
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles, tfidf_values, "o-", label="TF-IDF + LR")
    ax.fill(angles, tfidf_values, alpha=0.25)

    ax.plot(angles, bert_values, "o-", label="BERT")
    ax.fill(angles, bert_values, alpha=0.25)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title("模型性能雷达图")
    ax.legend(loc="best")
    st.pyplot(fig)
