# -*- coding: utf-8 -*-
"""フェイクニュース信頼度判定アプリ（SVM版 / TF-IDF + LinearSVC）"""

import os
import re

import numpy as np
import joblib
import fugashi
import streamlit as st
from scipy.sparse import hstack

# ──────────────────────────────────────────
# 保存済みファイルの読み込み
# ──────────────────────────────────────────
BASE = os.path.dirname(__file__)
ART = os.path.join(BASE, "svm_artifacts")

model         = joblib.load(os.path.join(ART, "svm_model.pkl"))
vectorizer    = joblib.load(os.path.join(ART, "vectorizer.pkl"))
scaler        = joblib.load(os.path.join(ART, "scaler.pkl"))
config        = joblib.load(os.path.join(ART, "config.pkl"))
feature_stats = joblib.load(os.path.join(ART, "feature_stats.pkl"))

FEATURE_COLUMNS = config["feature_columns"]

tagger = fugashi.Tagger()

# ──────────────────────────────────────────
# 分かち書き（学習時と同じ処理）
# ──────────────────────────────────────────
def tokenize_to_string(text):
    text = re.sub(r"(共同通信|NHK|読売新聞|毎日新聞|朝日新聞|産経新聞|中日新聞|神奈川新聞)", "", str(text))
    text = re.sub(r"によると?", "", text)
    text = re.sub(r"報道", "", text)

    tokens = []
    for word in tagger(text):
        pos = word.feature[0]
        surface = word.surface
        if (
            pos in ["名詞", "動詞", "形容詞"]
            and len(surface) > 1
            and not re.fullmatch(r'[0-9０-９]+', surface)
            and not re.fullmatch(r'[^\w぀-ヿ一-鿿]+', surface)
        ):
            tokens.append(surface)
    return " ".join(tokens)

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

def proper_noun_ratio(text):
    words = list(tagger(str(text)))
    total = len(words)
    proper = sum(
        1 for w in words
        if len(w.feature) > 1 and w.feature[0] == "名詞" and w.feature[1] == "固有名詞"
    )
    return proper / total if total > 0 else 0

def noun_ratio(text):
    words = list(tagger(str(text)))
    total = len(words)
    noun = sum(1 for w in words if w.feature[0] == "名詞")
    return noun / max(total, 1)

def extract_features(text):
    base = {
        "symbol_count":        symbol_count(text),
        "text_length":         text_length(text),
        "digit_ratio":         digit_ratio(text),
        "proper_noun_ratio":   proper_noun_ratio(text),
        "avg_sentence_length": avg_sentence_length(text),
        "noun_ratio":          noun_ratio(text),
        "sentence_count":      sentence_count(text),
        "kanji_ratio":         kanji_ratio(text),
        "hiragana_ratio":      hiragana_ratio(text),
        "report_style_count":  report_style_count(text),
        "quote_style_count":   quote_style_count(text),
        "person_info_count":   person_info_count(text),
    }
    return [base[col] for col in FEATURE_COLUMNS]

# ──────────────────────────────────────────
# 判定理由の説明文（多いとき, 少ないとき）
# ──────────────────────────────────────────
FEATURE_PHRASES = {
    "symbol_count":        ("記号が多めに使われています", "記号がほとんど使われていません"),
    "text_length":         ("文章が長めです", "文章が短めです"),
    "digit_ratio":         ("文章に占める数字の割合が高めです", "文章に占める数字の割合が低めです"),
    "proper_noun_ratio":   ("固有名詞（地名・組織名など）が多く含まれています", "固有名詞は少なめです"),
    "avg_sentence_length": ("一文が長めです", "一文が短めです"),
    "noun_ratio":          ("名詞の割合が高めです", "名詞の割合が低めです"),
    "sentence_count":      ("文の数が多めです", "文の数が少なめです"),
    "kanji_ratio":         ("漢字が多めに使われています", "漢字が少なめです"),
    "hiragana_ratio":      ("ひらがなが多めです", "ひらがなが少なめです"),
    "report_style_count":  ("「発表」「実施」などの報道的な表現が多く使われています", "報道的な表現はあまり使われていません"),
    "quote_style_count":   ("「〜によると」などソースを示す表現が多く使われています", "ソースを示す表現はあまり使われていません"),
    "person_info_count":   ("「〜氏」「〜さん」など人物に関する情報が多く含まれています", "人物に関する情報は少なめです"),
}

def make_reasons(feature_values):
    """各特徴量を、信頼度を上げる特徴／下げる特徴に分けて返す"""
    real_mean = feature_stats["real_mean"]
    fake_mean = feature_stats["fake_mean"]
    up, down = [], []
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
        if (val - mid) * gap > 0:
            down.append((strength, phrase))
        else:
            up.append((strength, phrase))
    up.sort(reverse=True)
    down.sort(reverse=True)
    return [p for _, p in up], [p for _, p in down]

# ──────────────────────────────────────────
# 予測
# ──────────────────────────────────────────
def predict_reliability(text):
    text = re.sub(r'\s+', '', text)

    # TF-IDF（学習時と同じ分かち書きをしてから変換）
    x_tfidf = vectorizer.transform([tokenize_to_string(text)])

    # 独自特徴量を標準化して結合（学習時と同じ）
    feats = extract_features(text)
    feats_scaled = scaler.transform(np.array(feats).reshape(1, -1))
    x = hstack((x_tfidf, feats_scaled))

    proba = model.predict_proba(x)[0]
    real_prob = proba[0]                      # クラス0 = Real
    reliability = round(real_prob * 100, 1)

    up, down = make_reasons(feats)
    return reliability, up, down

# ──────────────────────────────────────────
# 画面
# ──────────────────────────────────────────
st.title("ニュース信頼度の目安（SVM版）")
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
