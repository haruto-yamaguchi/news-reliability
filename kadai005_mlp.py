# -*- coding: utf-8 -*-
"""
フェイクニュース判定 - TF-IDF + MLP版
変更点: LinearSVC → MLPClassifier
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier          # ★ MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack

import fugashi
from fugashi import GenericTagger
from tqdm import tqdm

# ──────────────────────────────────────────
# 1. MeCab 初期化
# ──────────────────────────────────────────
tagger = GenericTagger(
    '-r "C:\\Program Files\\MeCab\\etc\\mecabrc" '
    '-d "C:\\Program Files\\MeCab\\dic\\ipadic"'
)

# ──────────────────────────────────────────
# 2. データ読み込み
# ──────────────────────────────────────────
df = pd.read_csv("fakenews_012new.csv", encoding="utf-8")
print(df.head())
print(df["isfake"].value_counts())

# ──────────────────────────────────────────
# 3. MeCab トークナイザ
# ──────────────────────────────────────────
STOP_WORDS = {"こと", "よう", "ため", "それ", "これ", "もの", "なっ", "れる", "られ"}

pbar = None

def mecab_tokenizer(text):
    text = re.sub(
        r"(共同通信|NHK|読売新聞|毎日新聞|朝日新聞|産経新聞|中日新聞|神奈川新聞)",
        "", str(text)
    )
    text = re.sub(r"によると?", "", text)
    text = re.sub(r"報道", "", text)

    if pbar is not None:
        pbar.update(1)

    tokens = []
    for word in tagger(text):
        pos     = word.feature[0]
        surface = word.surface
        if (
            pos in ["名詞", "動詞", "形容詞"]
            and len(surface) > 1
            and not re.fullmatch(r'[0-9０-９]+', surface)
            and not re.fullmatch(r'[^\w\u3040-\u30FF\u4E00-\u9FFF]+', surface)
            and surface not in STOP_WORDS
        ):
            tokens.append(surface)
    return tokens

# ──────────────────────────────────────────
# 4. 独自特徴量
# ──────────────────────────────────────────
def symbol_count(text):
    return len(re.findall(r'[^\w\u3040-\u30FF\u4E00-\u9FFF\s]', text))

def text_length(text):
    return len(text)

def digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)

def proper_noun_ratio(text):
    words = list(tagger(str(text)))
    total  = len(words)
    proper = sum(
        1 for w in words
        if len(w.feature) > 1 and w.feature[0] == "名詞" and w.feature[1] == "固有名詞"
    )
    return proper / total if total > 0 else 0

def avg_sentence_length(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    return sum(len(s) for s in sentences) / len(sentences) if sentences else 0

def noun_ratio(text):
    words = list(tagger(str(text)))
    return sum(1 for w in words if w.feature[0] == "名詞") / max(len(words), 1)

def sentence_count(text):
    return len([s for s in re.split(r'[。！？]', text) if s])

def kanji_ratio(text):
    return len(re.findall(r'[一-龯]', text)) / max(len(text), 1)

def hiragana_ratio(text):
    return len(re.findall(r'[ぁ-ん]', text)) / max(len(text), 1)

def report_style_count(text):
    words = ["発表","協議","協力","現地","今年","昨年","今後","今回",
             "行った","開か","実施","確認","報道"]
    return sum(text.count(w) for w in words)

def quote_style_count(text):
    words = ["によると","と語った","と述べた","と発表した","明らかにした",
             "関係者は","としている","とのこと","という"]
    return sum(text.count(w) for w in words)

def person_info_count(text):
    words = ["さん","氏","出身","卒業","語った","述べた","説明した"]
    return sum(text.count(w) for w in words)

feature_columns = [
    "symbol_count", "text_length", "digit_ratio",
    "proper_noun_ratio", "avg_sentence_length", "noun_ratio",
    "sentence_count", "kanji_ratio", "hiragana_ratio",
    "report_style_count", "quote_style_count", "person_info_count",
]

print("独自特徴量を計算中...")
df["symbol_count"]        = df["context"].apply(symbol_count)
df["text_length"]         = df["context"].apply(text_length)
df["digit_ratio"]         = df["context"].apply(digit_ratio)
df["proper_noun_ratio"]   = df["context"].apply(proper_noun_ratio)
df["avg_sentence_length"] = df["context"].apply(avg_sentence_length)
df["noun_ratio"]          = df["context"].apply(noun_ratio)
df["sentence_count"]      = df["context"].apply(sentence_count)
df["kanji_ratio"]         = df["context"].apply(kanji_ratio)
df["hiragana_ratio"]      = df["context"].apply(hiragana_ratio)
df["report_style_count"]  = df["context"].apply(report_style_count)
df["quote_style_count"]   = df["context"].apply(quote_style_count)
df["person_info_count"]   = df["context"].apply(person_info_count)

# ──────────────────────────────────────────
# 5. 学習 / テスト分割
# ──────────────────────────────────────────
x     = df["context"]
y     = df["isfake"]
extra = df[feature_columns]

x_train, x_test, y_train, y_test, extra_train, extra_test = train_test_split(
    x, y, extra, test_size=0.2, random_state=42
)

scaler             = StandardScaler()
extra_train_scaled = scaler.fit_transform(extra_train)
extra_test_scaled  = scaler.transform(extra_test)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ──────────────────────────────────────────
# 6. TF-IDF ベクトル化
# ──────────────────────────────────────────
pbar = tqdm(total=len(x_train), desc="MeCab処理（学習データ）")

vectorizer = TfidfVectorizer(
    max_features=10000,
    tokenizer=mecab_tokenizer,
    token_pattern=None,
    max_df=0.85,
    min_df=3,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
x_train_tfidf = vectorizer.fit_transform(x_train)
pbar.close()

pbar = tqdm(total=len(x_test), desc="MeCab処理（テストデータ）")
x_test_tfidf = vectorizer.transform(x_test)
pbar.close()
pbar = None

# TF-IDF + 独自特徴量を結合
x_train_final = hstack((x_train_tfidf, extra_train_scaled))
x_test_final  = hstack((x_test_tfidf,  extra_test_scaled))

print(f"最終特徴量 shape: {x_train_final.shape}")

# ──────────────────────────────────────────
# 7. ★ MLP モデル学習
# ──────────────────────────────────────────
model = MLPClassifier(
    # 層の構成（試しやすいように3パターンをコメントで記載）
    hidden_layer_sizes=(256, 128),   # 2層構成（デフォルト）
    # hidden_layer_sizes=(512, 256, 128),  # 3層（より複雑なパターンを学習）
    # hidden_layer_sizes=(256,),           # 1層（軽量・過学習しにくい）

    activation="relu",               # 活性化関数
    solver="adam",                   # 最適化アルゴリズム
    alpha=0.01,                     # L2正則化（大きいほど過学習を抑制）
    batch_size=256,                  # ミニバッチサイズ
    learning_rate_init=0.001,        # 初期学習率
    max_iter=200,                    # 最大エポック数
    early_stopping=True,             # ★過学習防止のため検証データで早期終了
    validation_fraction=0.1,         # 早期終了用の検証割合
    n_iter_no_change=10,             # n回改善なければ停止
    random_state=42,
    verbose=True,                    # 学習経過を表示
)

# fit時にclass_weightを渡す
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
print("\nMLP学習中...")
model.fit(x_train_final, y_train, sample_weight=sample_weights)  # ← weightを渡す


# model.fit(x_train_final, y_train)

# ──────────────────────────────────────────
# 8. 評価
# ──────────────────────────────────────────
y_pred   = model.predict(x_test_final)
proba    = model.predict_proba(x_test_final)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Accuracy: {accuracy:.4f} ===")

fake_score = proba[:, 1] + proba[:, 2]
actual_map = {0: "Real", 1: "Fake", 2: "Fake"}

print("\n--- サンプル判定結果 ---")
for i in range(5):
    label      = "Real" if y_pred[i] == 0 else "Fake"
    real_score = round((1 - fake_score[i]) * 100, 2)
    print(f"ニュース本文（先頭200字）:")
    print(x_test.iloc[i][:200])
    print(f"判定: {label}  信頼度: {real_score}%  正解: {actual_map[y_test.iloc[i]]}")
    print("-" * 50)

# 損失曲線（学習の推移を確認）
plt.figure(figsize=(8, 4))
plt.plot(model.loss_curve_, label="train loss")
if model.best_loss_ is not None:
    plt.axhline(model.best_loss_, linestyle="--", color="gray", label="best loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("MLP 学習曲線")
plt.legend()
plt.savefig("mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# 混同行列
cm = confusion_matrix(y_test, y_pred, normalize="true")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title(f"TF-IDF + MLP  Accuracy={accuracy:.4f}")
plt.savefig("confusion_matrix_mlp.png", dpi=150, bbox_inches="tight")
plt.show()
