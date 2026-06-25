# -*- coding: utf-8 -*-
"""フェイクニュース信頼度判定アプリ（TextCNN版）"""

import os
import re

import numpy as np
import joblib
import fugashi
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# ──────────────────────────────────────────
# 保存済みファイルの読み込み（app.pyがある場所を基準にする）
# ──────────────────────────────────────────
BASE = os.path.dirname(__file__)
ART = os.path.join(BASE, "cnn_artifacts")

config        = joblib.load(os.path.join(ART, "config.pkl"))
vocab         = joblib.load(os.path.join(ART, "vocab.pkl"))
scaler        = joblib.load(os.path.join(ART, "scaler.pkl"))
feature_stats = joblib.load(os.path.join(ART, "feature_stats.pkl"))

FEATURE_COLUMNS = config["feature_columns"]
MAX_LEN         = config["max_len"]

tagger = fugashi.Tagger()

# ──────────────────────────────────────────
# トークナイザ（学習時と同じ）
# ──────────────────────────────────────────
STOP_WORDS = {"こと", "よう", "ため", "それ", "これ", "もの", "なっ", "れる", "られ"}

def mecab_tokenizer(text):
    tokens = []
    for word in tagger(str(text)):
        pos = word.feature[0]
        if pos in ["名詞", "動詞", "形容詞"] and word.surface not in STOP_WORDS:
            tokens.append(word.surface)
    return tokens

def encode(text):
    tokens = mecab_tokenizer(text)
    ids = [vocab.get(token, 1) for token in tokens]
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

# ──────────────────────────────────────────
# 独自特徴量（学習時と同じ定義）
# ──────────────────────────────────────────
def symbol_count(text):
    return len(re.findall(r'[^\w぀-ヿ一-鿿\s]', text))

def text_length(text):
    return len(text)

def digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)

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

def mecab_features(text):
    """proper_noun_ratio / noun_ratio / lexical_diversity をまとめて計算"""
    words = list(tagger(str(text)))
    total = len(words)
    if total == 0:
        return {"proper_noun_ratio": 0.0, "noun_ratio": 0.0, "lexical_diversity": 0.0}
    surfaces = [w.surface for w in words]
    proper = sum(
        1 for w in words
        if len(w.feature) > 1 and w.feature[0] == "名詞" and w.feature[1] == "固有名詞"
    )
    noun = sum(1 for w in words if w.feature[0] == "名詞")
    return {
        "proper_noun_ratio": proper / total,
        "noun_ratio": noun / total,
        "lexical_diversity": len(set(surfaces)) / total,
    }

def extract_features(text):
    """feature_columns の並び順で特徴量ベクトルを返す"""
    base = {
        "symbol_count":        symbol_count(text),
        "text_length":         text_length(text),
        "digit_ratio":         digit_ratio(text),
        "avg_sentence_length": avg_sentence_length(text),
        "sentence_count":      sentence_count(text),
        "kanji_ratio":         kanji_ratio(text),
        "hiragana_ratio":      hiragana_ratio(text),
        "report_style_count":  report_style_count(text),
        "quote_style_count":   quote_style_count(text),
        "person_info_count":   person_info_count(text),
        "comma_count":         comma_count(text),
        "numeric_specificity": numeric_specificity(text),
        "comma_per_sentence":  comma_per_sentence(text),
    }
    base.update(mecab_features(text))
    return [base[col] for col in FEATURE_COLUMNS]

# ──────────────────────────────────────────
# 判定理由の言い回し（多いとき, 少ないとき）
# ──────────────────────────────────────────
# 各特徴量の説明文（多いとき, 少ないとき）
FEATURE_PHRASES = {
    "symbol_count":        ("記号が多めに使われています", "記号がほとんど使われていません"),
    "text_length":         ("文章が長めです", "文章が短めです"),
    "digit_ratio":         ("文章に占める数字の割合が高めです", "文章に占める数字の割合が低めです"),
    "avg_sentence_length": ("一文が長めです", "一文が短めです"),
    "sentence_count":      ("文の数が多めです", "文の数が少なめです"),
    "kanji_ratio":         ("漢字が多めに使われています", "漢字が少なめです"),
    "hiragana_ratio":      ("ひらがなが多めです", "ひらがなが少なめです"),
    "report_style_count":  ("「発表」「実施」などの報道的な表現が多く使われています", "報道的な表現はあまり使われていません"),
    "quote_style_count":   ("「〜によると」などソースを示す表現が多く使われています", "ソースを示す表現はあまり使われていません"),
    "person_info_count":   ("「〜氏」「〜さん」など人物に関する情報が多く含まれています", "人物に関する情報は少なめです"),
    "comma_count":         ("読点（、）が多めです", "読点（、）が少なめです"),
    "numeric_specificity": ("桁数の大きい具体的な数値が使われています", "具体的な数値は少なめです"),
    "comma_per_sentence":  ("一文あたりの読点が多めです", "一文あたりの読点が少なめです"),
    "proper_noun_ratio":   ("固有名詞（地名・組織名など）が多く含まれています", "固有名詞は少なめです"),
    "noun_ratio":          ("名詞の割合が高めです", "名詞の割合が低めです"),
    "lexical_diversity":   ("使われている語彙が多様です", "似た言葉が繰り返し使われています"),
}

def make_reasons(feature_values):
    """各特徴量を、信頼度を上げる特徴／下げる特徴に分けて返す"""
    real_mean = feature_stats["real_mean"]
    fake_mean = feature_stats["fake_mean"]
    up, down = [], []   # (強さ, 説明文)
    for col, val, r_m, f_m in zip(FEATURE_COLUMNS, feature_values, real_mean, fake_mean):
        mid = (r_m + f_m) / 2
        gap = f_m - r_m   # 正ならフェイク記事で多い特徴
        if abs(gap) < 1e-9:
            continue
        strength = abs(val - mid) / abs(gap)
        if strength < 0.25:
            continue
        high_phrase, low_phrase = FEATURE_PHRASES[col]
        phrase = high_phrase if val > mid else low_phrase
        # この値がフェイク記事側に寄っていれば信頼度を下げる、リアル側なら上げる
        if (val - mid) * gap > 0:
            down.append((strength, phrase))
        else:
            up.append((strength, phrase))
    up.sort(reverse=True)
    down.sort(reverse=True)
    return [p for _, p in up], [p for _, p in down]

# ──────────────────────────────────────────
# TextCNN モデル定義（学習時と同じ構造）
# ──────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, extra_dim, num_classes=2, embed_dim=128, num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, 3)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, 4)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, 5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters * 3 + extra_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, extra):
        x  = self.embedding(x)
        x  = x.transpose(1, 2)
        c3 = torch.max(F.relu(self.conv3(x)), dim=2)[0]
        c4 = torch.max(F.relu(self.conv4(x)), dim=2)[0]
        c5 = torch.max(F.relu(self.conv5(x)), dim=2)[0]
        x  = torch.cat([c3, c4, c5], dim=1)
        x  = self.dropout(x)
        x  = torch.cat([x, extra], dim=1)
        x  = F.relu(self.fc1(x))
        return self.fc2(x)

# モデルの読み込み（CPUで動かす）
@st.cache_resource
def load_model():
    m = TextCNN(
        vocab_size=config["vocab_size"],
        extra_dim=config["extra_dim"],
        num_classes=config["num_classes"],
    )
    state = torch.load(os.path.join(ART, "cnn_model.pt"), map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

model = load_model()

# ──────────────────────────────────────────
# 予測
# ──────────────────────────────────────────
def predict_reliability(text):
    text = re.sub(r'\s+', '', text)

    # テキストをID化
    ids = torch.tensor([encode(text)], dtype=torch.long)

    # 独自特徴量を標準化
    feats = extract_features(text)
    feats_scaled = scaler.transform(np.array(feats).reshape(1, -1)).astype(np.float32)
    extra = torch.tensor(feats_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(ids, extra)
        prob = F.softmax(output, dim=1)[0].numpy()

    real_prob = prob[0]                      # クラス0 = Real
    reliability = round(real_prob * 100, 1)  # 信頼度 = リアルである確率
    up, down = make_reasons(feats)
    return reliability, up, down

# ──────────────────────────────────────────
# 画面
# ──────────────────────────────────────────
st.title("ニュース信頼度の目安")
st.caption("文章の書き方の特徴をもとに、信頼度の目安を数値で表示します。"
           "内容が事実かどうかを判定するものではありません。")
text = st.text_area("ニュース本文を貼り付けてください", height=250)

if st.button("判定する"):
    if not text.strip():
        st.warning("テキストを入力してください")
    else:
        score, up, down = predict_reliability(text)
        st.metric("信頼度の目安", f"{score}%")

        if up:
            st.write("**信頼度を上げている特徴:**")
            for r in up:
                st.write("・", r)
        if down:
            st.write("**信頼度を下げている特徴:**")
            for r in down:
                st.write("・", r)
        if not up and not down:
            st.write("特筆すべき特徴は見られませんでした。")
