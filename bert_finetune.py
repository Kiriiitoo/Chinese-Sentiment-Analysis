# bert_finetune.py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import os

# ------------------------------
# 参数设置
# ------------------------------
MODEL_NAME = "bert-base-chinese"
DATA_PATH = "data/data_full.csv"   # 你可以换成 data_cut.csv
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 数据加载
# ------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ------------------------------
# 数据预处理
# ------------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

# 取少量数据调试（可删）
# df = df.sample(2000, random_state=42)

from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_ds = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_ds = SentimentDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ------------------------------
# 模型初始化
# ------------------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# ------------------------------
# 训练函数
# ------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ------------------------------
# 测试函数
# ------------------------------
def eval_model(model, loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    print("\nAccuracy:", acc)
    print(classification_report(labels_all, preds, digits=4))

# ------------------------------
# 主流程
# ------------------------------
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    loss = train_epoch(model, train_loader, optimizer)
    print(f"Training loss: {loss:.4f}")

print("\nEvaluating on test set...")
eval_model(model, test_loader)

# 保存模型
os.makedirs("model", exist_ok=True)
model.save_pretrained("model/bert_sentiment")
tokenizer.save_pretrained("model/bert_sentiment")
print("\n✅ 模型已保存到 model/bert_sentiment/")
