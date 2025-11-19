# baseline_tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

DATA = "data/data_cut.csv"

# 定义可序列化的分词函数
def my_tokenizer(x):
    return x.split()

def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=['text_cut', 'label'])
    df['label'] = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_cut'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    vect = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        tokenizer=my_tokenizer,
        preprocessor=None,
        token_pattern=None
    )
    X_tr = vect.fit_transform(X_train)
    X_te = vect.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)

    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, digits=4))

    os.makedirs("model", exist_ok=True)
    joblib.dump(vect, "model/tfidf_vectorizer.joblib")
    joblib.dump(clf, "model/logistic.joblib")
    print("模型已保存到 model/")

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
