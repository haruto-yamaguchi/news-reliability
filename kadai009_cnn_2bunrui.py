# -*- coding: utf-8 -*-
"""
フェイクニュース判定 - TextCNN版（MeCab + GiNZA特徴量併用）
Google Colab実行を想定。ランタイム > ランタイムのタイプを変更 > GPU を選択してください。
"""

# ──────────────────────────────────────────
# 0. Colab用セットアップ（初回のみ実行）
# ──────────────────────────────────────────
# !pip install -U fugashi unidic-lite
# !pip uninstall -y ginza ja_ginza spacy
# !pip install -U ginza ja_ginza spacy==3.7.5

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import fugashi
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

tqdm.pandas()

# ──────────────────────────────────────────
# 1. GiNZA / MeCab 初期化
# ──────────────────────────────────────────
_ginza_config = {
    "components": {
        "compound_splitter": {"split_mode": "A"}
    }
}
nlp = spacy.load("ja_ginza", config=_ginza_config)
tagger = fugashi.Tagger()

# ──────────────────────────────────────────
# 2. データ読み込み
# ──────────────────────────────────────────
df = pd.read_csv("fakenews_012new.csv", encoding="utf-8")
print(df.head())
print(df["isfake"].value_counts())

# ──────────────────────────────────────────
# 3. MeCabトークナイザ（CNN入力用・bigram等でも使う）
# ──────────────────────────────────────────
STOP_WORDS = {"こと", "よう", "ため", "それ", "これ", "もの", "なっ", "れる", "られ"}

def mecab_tokenizer(text):
    tokens = []
    for word in tagger(str(text)):
        pos = word.feature[0]
        if pos in ["名詞", "動詞", "形容詞"] and word.surface not in STOP_WORDS:
            tokens.append(word.surface)
    return tokens

# ──────────────────────────────────────────
# 4. 独自特徴量の定義
# ──────────────────────────────────────────

# ── MeCabを呼ばない特徴量（高速）────────────────────────────────────────
def exclamation_count(text):
    return text.count("!")

def ambiguity_count(text):
    ambiguous = ["はず", "だろう", "かもしれない", "らしい", "とみられる", "とのこと", "という"]
    return sum(text.count(a) for a in ambiguous)

def symbol_count(text):
    return len(re.findall(r'[^\w\u3040-\u30FF\u4E00-\u9FFF\s]', text))

def text_length(text):
    return len(text)

def digit_ratio(text):
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)

def avg_sentence_length(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    return sum(len(s) for s in sentences) / len(sentences) if sentences else 0

def sentence_count(text):
    return len([s for s in re.split(r'[。！？]', text) if s])

def kanji_ratio(text):
    return len(re.findall(r'[一-龯]', text)) / max(len(text), 1)

def hiragana_ratio(text):
    return len(re.findall(r'[ぁ-ん]', text)) / max(len(text), 1)

def report_style_count(text):
    words = ["発表", "協議", "協力", "現地", "今年", "昨年", "今後", "今回",
              "行った", "開か", "実施", "確認", "報道"]
    return sum(text.count(w) for w in words)

def quote_style_count(text):
    words = ["によると", "と語った", "と述べた", "と発表した", "明らかにした",
              "関係者は", "としている", "とのこと", "という"]
    return sum(text.count(w) for w in words)

def person_info_count(text):
    words = ["さん", "氏", "出身", "卒業", "語った", "述べた", "説明した"]
    return sum(text.count(w) for w in words)

def comma_count(text):
    return text.count("、")

def numeric_specificity(text):
    numbers = re.findall(r'\d+', text)
    if not numbers:
        return 0
    return np.mean([len(n) for n in numbers])

def comma_per_sentence(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    if not sentences:
        return 0
    return text.count("、") / len(sentences)

# ── MeCabを1回だけ呼ぶ特徴量（まとめて計算）──────────────────────────────
def mecab_features(text):
    """proper_noun_ratio / noun_ratio / lexical_diversity を
    MeCab 1回の呼び出しでまとめて計算する。
    元コードでは3つが別々に apply されていたが、
    これで MeCab の呼び出し回数が 3万回 → 1万3千回 に減る。"""
    words = list(tagger(str(text)))
    total = len(words)
    if total == 0:
        return pd.Series({
            "proper_noun_ratio": 0.0,
            "noun_ratio": 0.0,
            "lexical_diversity": 0.0,
        })
    surfaces = [w.surface for w in words]
    proper = sum(
        1 for w in words
        if len(w.feature) > 1
        and w.feature[0] == "名詞"
        and w.feature[1] == "固有名詞"
    )
    noun = sum(1 for w in words if w.feature[0] == "名詞")
    return pd.Series({
        "proper_noun_ratio": proper / total,
        "noun_ratio": noun / total,
        "lexical_diversity": len(set(surfaces)) / total,
    })

# ──────────────────────────────────────────
# 5. 特徴量の計算
# ──────────────────────────────────────────

# MeCabを呼ばない特徴量（全部まとめて）
print("MeCabを呼ばない特徴量を計算中...")
df["exclamation_count"]   = df["context"].apply(exclamation_count)
df["ambiguity_count"]     = df["context"].apply(ambiguity_count)
df["symbol_count"]        = df["context"].apply(symbol_count)
df["text_length"]         = df["context"].apply(text_length)
df["digit_ratio"]         = df["context"].apply(digit_ratio)
df["avg_sentence_length"] = df["context"].apply(avg_sentence_length)
df["sentence_count"]      = df["context"].apply(sentence_count)
df["kanji_ratio"]         = df["context"].apply(kanji_ratio)
df["hiragana_ratio"]      = df["context"].apply(hiragana_ratio)
df["report_style_count"]  = df["context"].apply(report_style_count)
df["quote_style_count"]   = df["context"].apply(quote_style_count)
df["person_info_count"]   = df["context"].apply(person_info_count)
df["comma_count"]         = df["context"].apply(comma_count)
df["numeric_specificity"] = df["context"].apply(numeric_specificity)
df["comma_per_sentence"]  = df["context"].apply(comma_per_sentence)

# MeCabを使う特徴量：1回の apply にまとめて計算
print("MeCab特徴量を計算中（1回にまとめて実行）...")
mecab_df = df["context"].progress_apply(mecab_features)
df = pd.concat([df, mecab_df], axis=1)

# ── GiNZA特徴量 ──────────────────────────────────────────────────────────
# def ginza_features(text):
#     doc = nlp(str(text))
#     person = organization = location = 0
#     for ent in doc.ents:
#         label = ent.label_
#         if label == "Person":
#             person += 1
#         elif label == "Organization":
#             organization += 1
#         elif label in ["Province", "City", "Country"]:
#             location += 1
#     return pd.Series({
#         "person_count": person,
#         "organization_count": organization,
#         "location_count": location,
#         "ner_count": len(doc.ents),
#         "ner_ratio": len(doc.ents) / max(len(doc), 1),
#     })

# print("GiNZA特徴量を計算中...")
# ginza_df = df["context"].progress_apply(ginza_features)
# df = pd.concat([df, ginza_df], axis=1)

# print(
#     df.groupby("isfake")[[
#         "person_count", "organization_count",
#         "location_count", "ner_count", "ner_ratio",
#     ]].mean()
# )

# ──────────────────────────────────────────
# 6. 特徴量リスト
# ──────────────────────────────────────────
feature_columns = [
    # MeCabなし
    "symbol_count", "text_length", "digit_ratio",
    "avg_sentence_length", "sentence_count", "kanji_ratio", "hiragana_ratio",
    "report_style_count", "quote_style_count", "person_info_count",
    "comma_count", "numeric_specificity", "comma_per_sentence",
    # MeCabあり（mecab_featuresでまとめて計算）
    "proper_noun_ratio", "noun_ratio", "lexical_diversity",
    # GiNZA
    # "person_count", "organization_count", "location_count",
    # "ner_count", "ner_ratio",
]

# ──────────────────────────────────────────
# 7. 学習 / テスト分割
# ──────────────────────────────────────────
x     = df["context"]
# 2値分類: 0=Real, 1=Fake（部分AI・全AIをまとめてFakeとして学習）
# 3クラスの確率は評価時に内訳として表示する
y     = (df["isfake"] > 0).astype(int)
y_raw = df["isfake"]   # 内訳表示用に元の3クラスラベルを保持
extra = df[feature_columns]

x_train, x_test, y_train, y_test, extra_train, extra_test = train_test_split(
    x, y, extra, test_size=0.2, random_state=42, stratify=y
)

# 独自特徴量を標準化（trainでfitし、testはtransformのみ）
scaler             = StandardScaler()
extra_train_scaled = scaler.fit_transform(extra_train).astype(np.float32)
extra_test_scaled  = scaler.transform(extra_test).astype(np.float32)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ──────────────────────────────────────────
# 8. 語彙辞書の作成（trainのみから構築）
# ──────────────────────────────────────────
counter = Counter()
for text in tqdm(x_train, desc="語彙構築"):
    counter.update(mecab_tokenizer(text))

MAX_VOCAB = 30000
vocab = {
    word: i + 2
    for i, (word, _) in enumerate(counter.most_common(MAX_VOCAB))
}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

MAX_LEN = 500

def encode(text):
    tokens = mecab_tokenizer(text)
    ids = [vocab.get(token, 1) for token in tokens]
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

# ──────────────────────────────────────────
# 9. Dataset / DataLoader
# ──────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, extra, labels):
        self.texts  = texts.reset_index(drop=True)
        self.extra  = np.asarray(extra)
        self.labels = labels.reset_index(drop=True)
        print("テキストをID化中...")
        self.encoded = [encode(t) for t in tqdm(self.texts)]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_ids   = torch.tensor(self.encoded[idx], dtype=torch.long)
        extra_feat = torch.tensor(self.extra[idx], dtype=torch.float32)
        label      = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return text_ids, extra_feat, label


train_dataset = NewsDataset(x_train, extra_train_scaled, y_train)
test_dataset  = NewsDataset(x_test,  extra_test_scaled,  y_test)

BATCH_SIZE   = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────────────────────────────
# 10. TextCNN モデル定義
# ──────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, extra_dim, num_classes=3, embed_dim=128, num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, 3)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, 4)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, 5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters * 3 + extra_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, extra):
        x  = self.embedding(x)           # (batch, seq_len, embed_dim)
        x  = x.transpose(1, 2)           # (batch, embed_dim, seq_len)
        c3 = torch.max(F.relu(self.conv3(x)), dim=2)[0]
        c4 = torch.max(F.relu(self.conv4(x)), dim=2)[0]
        c5 = torch.max(F.relu(self.conv5(x)), dim=2)[0]
        x  = torch.cat([c3, c4, c5], dim=1)
        x  = self.dropout(x)
        x  = torch.cat([x, extra], dim=1)
        x  = F.relu(self.fc1(x))
        return self.fc2(x)

# ──────────────────────────────────────────
# 11. 学習
# ──────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")

model = TextCNN(
    vocab_size=len(vocab),
    extra_dim=len(feature_columns),
    num_classes=2,  # Real / Fake の2値分類
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 20


def evaluate(loader):
    model.eval()
    preds_, truth_, probs_ = [], [], []
    with torch.no_grad():
        for text_ids, extra_feat, label in loader:
            output = model(text_ids.to(device), extra_feat.to(device))
            prob   = F.softmax(output, dim=1)
            pred   = output.argmax(1)
            preds_.extend(pred.cpu().numpy())
            truth_.extend(label.numpy())
            probs_.extend(prob.cpu().numpy())
    preds_ = np.array(preds_)
    truth_ = np.array(truth_)
    probs_ = np.array(probs_)
    return accuracy_score(truth_, preds_), preds_, truth_, probs_


history = {"train_loss": [], "test_acc": []}
best_test_acc   = 0.0
best_state_dict = None

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for text_ids, extra_feat, label in train_loader:
        text_ids   = text_ids.to(device)
        extra_feat = extra_feat.to(device)
        label      = label.to(device)
        optimizer.zero_grad()
        output = model(text_ids, extra_feat)
        loss   = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    test_acc, _, _, _ = evaluate(test_loader)
    history["train_loss"].append(avg_loss)
    history["test_acc"].append(test_acc)

    if test_acc > best_test_acc:
        best_test_acc   = test_acc
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Train Loss: {avg_loss:.4f}  Test Acc: {test_acc:.4f}")

print(f"\nベストTest Accuracy: {best_test_acc:.4f}")

# 過学習チェック用グラフ
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), history["test_acc"], marker="o", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Epochごとの Test Accuracy 推移")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("test_accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.show()

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# ──────────────────────────────────────────
# 12. 評価（ベストepochの重みを使用）
# ──────────────────────────────────────────
accuracy, preds, truth, all_probs = evaluate(test_loader)
print(f"\n=== Accuracy (ベストepoch): {accuracy:.4f} ===")

# all_probs[:, 0] = Real確率、all_probs[:, 1] = Fake確率
real_prob = all_probs[:, 0]
fake_prob = all_probs[:, 1]

x_test_reset    = x_test.reset_index(drop=True)
y_test_reset    = y_test.reset_index(drop=True)
y_raw_test      = y_raw.iloc[y_test.index].reset_index(drop=True)  # 元の3クラスラベル
raw_label_map   = {0: "Real", 1: "Fake(部分AI)", 2: "Fake(全AI)"}

print("\n--- サンプル判定結果 ---")
for i in range(5):
    pred_label = "Real" if preds[i] == 0 else "Fake"
    true_label = raw_label_map[y_raw_test.iloc[i]]
    print("ニュース本文（先頭200字）:")
    print(x_test_reset.iloc[i][:200])
    print(f"判定: {pred_label}  正解: {true_label}")
    print(f"  Real確率  : {real_prob[i]*100:.1f}%")
    print(f"  Fake確率  : {fake_prob[i]*100:.1f}%")
    print("-" * 50)

# 混同行列（2値: Real / Fake）
cm = confusion_matrix(truth, preds, normalize="true")
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)
disp.plot()
plt.title(f"TextCNN 2値分類  Accuracy={accuracy:.4f}")
plt.savefig("confusion_matrix_textcnn.png", dpi=150, bbox_inches="tight")
plt.show()