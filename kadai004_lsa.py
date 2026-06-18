# -*- coding: utf-8 -*-
"""
フェイクニュース判定 - LSA（潜在意味解析）+ LinearSVC版
変更点: TF-IDFベクトルをTruncatedSVD（LSA）で次元削減してから学習
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD          # ★ LSAの本体
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack, csr_matrix

import fugashi
from fugashi import GenericTagger
from tqdm import tqdm

# ──────────────────────────────────────────
# 1. MeCab 初期化（Windows ローカル環境）
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
pbar = None  # train時に差し替え

STOP_WORDS = {"こと", "よう", "ため", "それ", "これ", "もの", "なっ", "れる", "られ"}  # ★追加

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
def exclamation_count(text):
    return text.count("!")

def ambiguity_count(text):
    ambiguous = ["はず","だろう","かもしれない","らしい","とみられる","とのこと","という"]
    return sum(text.count(a) for a in ambiguous)

def symbol_count(text):
    return len(re.findall(r'[^\w\u3040-\u30FF\u4E00-\u9FFF\s]', text))

def text_length(text):
    return len(text)

def digit_ratio(text):
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)

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
    total = len(words)
    noun  = sum(1 for w in words if w.feature[0] == "名詞")
    return noun / max(total, 1)

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

# 語数の豊富さ
def lexical_diversity(text):
    words = [w.surface for w in tagger(str(text))]
    return len(set(words)) / max(len(words),1)

# 、の数
def comma_count(text):
    return text.count("、")

# 独自特徴量を一括計算
feature_columns = [
    "symbol_count", "text_length", "digit_ratio",
    "proper_noun_ratio", "avg_sentence_length", "noun_ratio",
    "sentence_count", "kanji_ratio", "hiragana_ratio",
    "report_style_count", "quote_style_count", "person_info_count",
    "lexical_diversity", "comma_count",
]

print("独自特徴量を計算中...")
df["exclamation_count"]  = df["context"].apply(exclamation_count)
df["ambiguity_count"]    = df["context"].apply(ambiguity_count)
df["symbol_count"]       = df["context"].apply(symbol_count)
df["text_length"]        = df["context"].apply(text_length)
df["digit_ratio"]        = df["context"].apply(digit_ratio)
df["proper_noun_ratio"]  = df["context"].apply(proper_noun_ratio)
df["avg_sentence_length"]= df["context"].apply(avg_sentence_length)
df["noun_ratio"]         = df["context"].apply(noun_ratio)
df["sentence_count"]     = df["context"].apply(sentence_count)
df["kanji_ratio"]        = df["context"].apply(kanji_ratio)
df["hiragana_ratio"]     = df["context"].apply(hiragana_ratio)
df["report_style_count"] = df["context"].apply(report_style_count)
df["quote_style_count"]  = df["context"].apply(quote_style_count)
df["person_info_count"]  = df["context"].apply(person_info_count)
df["lexical_diversity"] = df["context"].apply(lexical_diversity)
df["comma_count"] = df["context"].apply(comma_count)
# ──────────────────────────────────────────
# 5. 学習 / テスト分割
# ──────────────────────────────────────────
x     = df["context"]
y     = df["isfake"]
extra = df[feature_columns]

x_train, x_test, y_train, y_test, extra_train, extra_test = train_test_split(
    x, y, extra, test_size=0.2, random_state=42
)

# 独自特徴量を標準化
scaler             = StandardScaler()
extra_train_scaled = scaler.fit_transform(extra_train)
extra_test_scaled  = scaler.transform(extra_test)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ──────────────────────────────────────────
# 6. TF-IDF ベクトル化
# ──────────────────────────────────────────
pbar = tqdm(total=len(x_train), desc="MeCab処理（学習データ）")

vectorizer = TfidfVectorizer(
    max_features=20000,
    tokenizer=mecab_tokenizer,
    token_pattern=None,
    max_df=0.85,
    min_df=4,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
x_train_tfidf = vectorizer.fit_transform(x_train)
pbar.close()

pbar = tqdm(total=len(x_test), desc="MeCab処理（テストデータ）")
x_test_tfidf = vectorizer.transform(x_test)
pbar.close()
pbar = None

print(f"TF-IDF shape: {x_train_tfidf.shape}")

# ──────────────────────────────────────────
# 7. ★ LSA（TruncatedSVD）で次元削減
# ──────────────────────────────────────────
# n_components: 削減後の次元数（100〜500 が一般的。大きいほど情報量UP・重くなる）
N_COMPONENTS = 2000

print(f"LSA: TF-IDF {x_train_tfidf.shape[1]}次元 → {N_COMPONENTS}次元に削減中...")

svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
x_train_lsa = svd.fit_transform(x_train_tfidf)   # shape: (n_train, 300)
x_test_lsa  = svd.transform(x_test_tfidf)         # shape: (n_test,  300)

# 寄与率の確認
explained = svd.explained_variance_ratio_.sum()
print(f"LSA 累積寄与率（{N_COMPONENTS}次元）: {explained:.3f}")

# LSA後にL2正規化（SVMの性能向上に有効）
normalizer      = Normalizer(copy=False)
x_train_lsa_n   = normalizer.fit_transform(x_train_lsa)
x_test_lsa_n    = normalizer.transform(x_test_lsa)

# ──────────────────────────────────────────
# 8. LSA + 独自特徴量を結合
# ──────────────────────────────────────────
x_train_final = np.hstack([x_train_lsa_n, extra_train_scaled])
x_test_final  = np.hstack([x_test_lsa_n,  extra_test_scaled])

print(f"最終特徴量 shape: {x_train_final.shape}")

# ──────────────────────────────────────────
# 9. モデル学習
# ──────────────────────────────────────────
base_model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=20000,
    C=1,
)
model = CalibratedClassifierCV(base_model, cv=3)
model.fit(x_train_final, y_train)

# ──────────────────────────────────────────
# 10. 評価
# ──────────────────────────────────────────
y_pred   = model.predict(x_test_final)
proba    = model.predict_proba(x_test_final)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Accuracy: {accuracy:.4f} ===")

# クラス確認
print("クラス:", model.classes_)
fake_score = proba[:, 1] + proba[:, 2]  # クラス 1, 2 が Fake

actual_map = {0: "Real", 1: "Fake", 2: "Fake"}

print("\n--- サンプル判定結果 ---")
for i in range(5):
    label      = "Real" if y_pred[i] == 0 else "Fake"
    real_score = round((1 - fake_score[i]) * 100, 2)
    print(f"ニュース本文（先頭200字）:")
    print(x_test.iloc[i][:200])
    print(f"判定: {label}  信頼度: {real_score}%  正解: {actual_map[y_test.iloc[i]]}")
    print("-" * 50)

# 混同行列
cm = confusion_matrix(y_test, y_pred, normalize="true")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title(f"LSA ({N_COMPONENTS}dim) + LinearSVC  Accuracy={accuracy:.4f}")
plt.savefig("confusion_matrix_lsa.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────
# 11. LSA トピック上位語の表示（参考）
# ──────────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
print("\n--- LSA 上位トピック（先頭5件）---")
for i, comp in enumerate(svd.components_[:5]):
    top_words = [feature_names[j] for j in np.argsort(comp)[-10:][::-1]]
    print(f"Topic {i}: {' / '.join(top_words)}")
