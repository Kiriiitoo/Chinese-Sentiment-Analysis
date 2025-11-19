# preprocess.py
import jieba
import pandas as pd
import sys

IN = "data/data_full.csv"  # 如果换数据，修改路径
OUT = "data/data_cut.csv"

def cut(text):
    # 基本分词：你可以在这里添加 jieba.add_word("某词") 强制识别
    return " ".join(jieba.lcut(str(text).strip()))

def main():
    df = pd.read_csv(IN)
    # 可选清洗：删除空文本、去除过短文本
    df = df.dropna(subset=['text'])
    df['text_cut'] = df['text'].apply(cut)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print("Saved to", OUT, "rows:", len(df))

if __name__ == "__main__":
    main()
