# -*- coding: utf-8 -*-
"""
フェイクニュース判定 - 特徴量分布の可視化のみ（学習なし）
Google Colab実行を想定。

このスクリプトは、独自特徴量(symbol_count等)とGiNZA特徴量(person_count等)を
計算し、クラス(Real/Fake部分AI/Fake全AI)ごとのKDE分布を重ねて表示するだけ。
TextCNNの学習・DataLoader・PyTorch関連は一切含まない。

特徴量の判別力を学習前に確認したいときに使う。
"""

# ──────────────────────────────────────────
# 0. Colab用セットアップ（初回のみ実行）
# ──────────────────────────────────────────
# !pip install -U ginza ja_ginza fugashi unidic-lite
# !pip install -q japanize-matplotlib

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
import spacy
import fugashi
from tqdm import tqdm

tqdm.pandas()

# Colabは日本語フォントが未インストールのため文字化け(豆腐文字)する。
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    print("日本語フォント未対応: `!pip install -q japanize-matplotlib` を実行してから"
          "再度importしてください。")

# ──────────────────────────────────────────
# 1. GiNZA / MeCab 初期化
# ──────────────────────────────────────────
# spaCy 3.8.x + ja_ginza 5.2.0 の組み合わせで compound_splitter が
# ConfigValidationError を起こすため config を明示的に渡して回避する。
_ginza_config = {
    "components": {
        "compound_splitter": {"split_mode": "A"}
    }
}
nlp = spacy.load("ja_ginza", config=_ginza_config)

# Colab(Linux)では unidic-lite を使うため fugashi.Tagger() を使う。
# （GenericTagger() はシステムのMeCabを探しに行くため Colab では RuntimeError になる）
tagger = fugashi.Tagger()

# ──────────────────────────────────────────
# 2. データ読み込み
# ──────────────────────────────────────────
df = pd.read_csv("fakenews.csv", encoding="utf-8")
print(df.head())
print(df["isfake"].value_counts())

# ──────────────────────────────────────────
# 3. MeCabトークナイザ
#    （bigram/trigram/topic_consistency などで使う共通関数）
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

# ── MeCabを呼ばない特徴量（高速）──────────────────
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

def punctuation_rhythm_std(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    if len(sentences) < 2:
        return 0
    return np.std([len(s) for s in sentences])

def comma_per_sentence(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    if not sentences:
        return 0
    return text.count("、") / len(sentences)

def conjunction_count(text):
    conjunctions = ["しかし", "また", "さらに", "一方", "そして", "なお",
                     "ただし", "つまり", "このように", "そのため"]
    return sum(text.count(c) for c in conjunctions)

def conjunction_diversity(text):
    conjunctions = ["しかし", "また", "さらに", "一方", "そして", "なお",
                     "ただし", "つまり", "このように", "そのため"]
    return len(set(c for c in conjunctions if c in text))

def sentence_ending_diversity(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    endings = [s[-2:] for s in sentences if len(s) >= 2]
    if not endings:
        return 0
    return len(set(endings)) / len(endings)

def desu_masu_ratio(text):
    desu_masu = len(re.findall(r'(です|ます)[。！？]', text))
    sc = len([s for s in re.split(r'[。！？]', text) if s])
    return desu_masu / max(sc, 1)

def date_expression_count(text):
    patterns = [r'\d+年', r'\d+月', r'\d+日', r'\d+時']
    return sum(len(re.findall(p, text)) for p in patterns)

def numeric_specificity(text):
    numbers = re.findall(r'\d+', text)
    if not numbers:
        return 0
    return np.mean([len(n) for n in numbers])

def round_number_ratio(text):
    numbers = re.findall(r'\d+', text)
    if not numbers:
        return 0
    round_nums = [n for n in numbers if int(n) % 10 == 0 and len(n) >= 2]
    return len(round_nums) / len(numbers)

def url_count(text):
    return len(re.findall(r'https?://', text))

def evaluative_expression_count(text):
    words = ["重要", "深刻", "懸念", "批判", "支持", "反発",
             "歓迎", "警戒", "注目", "問題視"]
    return sum(text.count(w) for w in words)

def intensifier_count(text):
    words = ["非常に", "極めて", "著しく", "大幅に", "急速に",
             "劇的に", "圧倒的に", "絶対に", "完全に"]
    return sum(text.count(w) for w in words)

def bigram_repetition_rate(text):
    tokens = mecab_tokenizer(text)
    if len(tokens) < 2:
        return 0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return 1 - (len(set(bigrams)) / len(bigrams))

def trigram_repetition_rate(text):
    tokens = mecab_tokenizer(text)
    if len(tokens) < 3:
        return 0
    trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    return 1 - (len(set(trigrams)) / len(trigrams))

def topic_consistency(text):
    sentences = [s for s in re.split(r'[。！？]', text) if s]
    if len(sentences) < 2:
        return 0
    first = set(mecab_tokenizer(sentences[0]))
    last  = set(mecab_tokenizer(sentences[-1]))
    union = first | last
    return len(first & last) / len(union) if union else 0


# ── MeCabを呼ぶ特徴量：1回の apply にまとめて計算（速度改善）──────────
def mecab_features(text):
    """proper_noun_ratio / noun_ratio / lexical_diversity を
    MeCab1回の呼び出しでまとめて計算する。"""
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
print("MeCabを呼ばない特徴量を計算中...")
df["exclamation_count"]          = df["context"].apply(exclamation_count)
df["ambiguity_count"]            = df["context"].apply(ambiguity_count)
df["symbol_count"]               = df["context"].apply(symbol_count)
df["text_length"]                = df["context"].apply(text_length)
df["digit_ratio"]                = df["context"].apply(digit_ratio)
df["avg_sentence_length"]        = df["context"].apply(avg_sentence_length)
df["sentence_count"]             = df["context"].apply(sentence_count)
df["kanji_ratio"]                = df["context"].apply(kanji_ratio)
df["hiragana_ratio"]             = df["context"].apply(hiragana_ratio)
df["report_style_count"]         = df["context"].apply(report_style_count)
df["quote_style_count"]          = df["context"].apply(quote_style_count)
df["person_info_count"]          = df["context"].apply(person_info_count)
df["comma_count"]                = df["context"].apply(comma_count)
df["punctuation_rhythm_std"]     = df["context"].apply(punctuation_rhythm_std)
df["comma_per_sentence"]         = df["context"].apply(comma_per_sentence)
df["conjunction_count"]          = df["context"].apply(conjunction_count)
df["conjunction_diversity"]      = df["context"].apply(conjunction_diversity)
df["sentence_ending_diversity"]  = df["context"].apply(sentence_ending_diversity)
df["desu_masu_ratio"]            = df["context"].apply(desu_masu_ratio)
df["date_expression_count"]      = df["context"].apply(date_expression_count)
df["numeric_specificity"]        = df["context"].apply(numeric_specificity)
df["round_number_ratio"]         = df["context"].apply(round_number_ratio)
df["url_count"]                  = df["context"].apply(url_count)
df["evaluative_expression_count"]= df["context"].apply(evaluative_expression_count)
df["intensifier_count"]          = df["context"].apply(intensifier_count)

# MeCabを使う特徴量：1回の apply でまとめて計算
print("MeCab特徴量を計算中（1回にまとめて実行）...")
mecab_df = df["context"].progress_apply(mecab_features)
df = pd.concat([df, mecab_df], axis=1)

# mecab_tokenizer を使う特徴量（MeCabを再度呼ぶため別グループ）
print("MeCabトークナイザを使う特徴量を計算中...")
df["bigram_repetition_rate"]  = df["context"].progress_apply(bigram_repetition_rate)
df["trigram_repetition_rate"] = df["context"].progress_apply(trigram_repetition_rate)
df["topic_consistency"]       = df["context"].progress_apply(topic_consistency)

# ── GiNZA特徴量（重いので必要なときだけコメントアウトを外す）────────────
def ginza_features(text):
    doc = nlp(str(text))
    person = organization = location = 0
    for ent in doc.ents:
        label = ent.label_
        if label == "Person":
            person += 1
        elif label == "Organization":
            organization += 1
        elif label in ["Province", "City", "Country"]:
            location += 1
    return pd.Series({
        "person_count": person,
        "organization_count": organization,
        "location_count": location,
        "ner_count": len(doc.ents),
        "ner_ratio": len(doc.ents) / max(len(doc), 1),
    })

# print("GiNZA特徴量を計算中...")
# ginza_df = df["context"].progress_apply(ginza_features)
# df = pd.concat([df, ginza_df], axis=1)

# ──────────────────────────────────────────
# 6. 可視化対象の特徴量リスト
# ──────────────────────────────────────────
feature_columns = [
    # MeCabなし
    "symbol_count", "text_length", "digit_ratio",
    "avg_sentence_length", "sentence_count", "kanji_ratio", "hiragana_ratio",
    "report_style_count", "quote_style_count", "person_info_count",
    "comma_count", "punctuation_rhythm_std", "comma_per_sentence",
    "conjunction_count", "conjunction_diversity", "sentence_ending_diversity",
    "desu_masu_ratio", "date_expression_count", "numeric_specificity",
    "round_number_ratio", "url_count",
    "evaluative_expression_count", "intensifier_count",
    # MeCabあり
    "proper_noun_ratio", "noun_ratio", "lexical_diversity",
    "bigram_repetition_rate", "trigram_repetition_rate", "topic_consistency",
    # GiNZA（使う場合はコメントアウトを外す）
    # "person_count", "organization_count", "location_count",
    # "ner_count", "ner_ratio",
]

viz_columns = feature_columns + ["exclamation_count", "ambiguity_count"]

print("\n--- クラスごとの平均値 ---")
print(df.groupby("isfake")[viz_columns].mean().T)

# ──────────────────────────────────────────
# 7. z-score標準化してKDE + rugplot（3行）で可視化
# ──────────────────────────────────────────
df_viz = df[viz_columns].copy()
col_std = df_viz.std().replace(0, 1)
df_viz  = (df_viz - df_viz.mean()) / col_std
df_viz["isfake"] = df["isfake"].values

class_labels = {0: "Real", 1: "Fake(部分AI)", 2: "Fake(全AI)"}
class_colors  = {0: "#2E86AB", 1: "#F6A623", 2: "#D7263D"}

# rugplot の各行の位置（axes座標 0〜1）
rug_bottom = {0: 0.16, 1: 0.08, 2: 0.00}
rug_top    = {0: 0.22, 1: 0.14, 2: 0.06}
RUG_SAMPLE = 200

n_cols = 4
n_rows = int(np.ceil(len(viz_columns) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(viz_columns):
    ax = axes[i]
    for cls, label in class_labels.items():
        subset = df_viz.loc[df_viz["isfake"] == cls, col]
        color  = class_colors[cls]

        # KDE（全件）
        if subset.nunique() > 1:
            subset.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)

        # rugplot：クラスごとに3行に分けて描画
        # x=data座標、y=axes座標の混合変換を使ってy位置を固定する
        sample = subset.sample(min(RUG_SAMPLE, len(subset)), random_state=42)
        trans  = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(
            x=sample.values,
            ymin=rug_bottom[cls],
            ymax=rug_top[cls],
            transform=trans,
            color=color,
            alpha=0.4,
            linewidth=0.8,
        )

    ax.set_title(col, fontsize=10)
    ax.set_xlabel("z-score", fontsize=7)
    ax.set_ylabel("")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlim(-6, 6)
    ax.legend(fontsize=7)

for j in range(len(viz_columns), len(axes)):
    axes[j].axis("off")

plt.suptitle("特徴量のクラス別KDE分布（z-score標準化済み）", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("feature_distributions_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
