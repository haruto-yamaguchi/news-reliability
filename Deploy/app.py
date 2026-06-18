import os
import joblib
import fugashi
import re
import numpy as np
from scipy.sparse import hstack
import streamlit as st

# モデルの読み込み（app.pyがある場所を基準にする）
BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, "models/kadai003_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "models/kadai003_vectorizer.pkl"))
scaler = joblib.load(os.path.join(BASE, "models/kadai003_scaler.pkl"))
real_stats = joblib.load(os.path.join(BASE, "models/kadai003_real_stats.pkl"))

tagger = fugashi.Tagger()

POSITIVE_WORDS = [
    "良い", "素晴らしい", "正確", "安全", "信頼", "確認", "事実", "公式",
    "成功", "改善", "発展", "解決", "安心", "正式", "明確", "適切"
]
NEGATIVE_WORDS = [
    "悪い", "危険", "嘘", "偽", "疑惑", "問題", "失敗", "不正",
    "批判", "衝撃", "炎上", "拡散", "デマ", "煽り", "怪しい", "不明"
]

def tokenize_text(text):
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

def extract_features(text):
    exclamation  = text.count("!")
    ambiguity    = sum(text.count(a) for a in ["かも", "思われる", "らしい", "のよう", "いわれている", "可能性", "だろう"])
    symbol       = len(re.findall(r'[^\w぀-ヿ一-鿿\s]', text))
    length       = len(text)
    digit_r      = sum(c.isdigit() for c in text) / max(len(text), 1)
    words        = list(tagger(str(text)))
    total        = len(words)
    proper       = sum(1 for w in words if len(w.feature) > 1 and w.feature[0] == "名詞" and w.feature[1] == "固有名詞")
    proper_r     = proper / total if total > 0 else 0
    sentences    = [s for s in re.split(r'[。！？]', text) if s]
    avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
    noun         = sum(1 for w in words if w.feature[0] == "名詞")
    noun_r       = noun / max(total, 1)
    sent_count   = len(sentences)
    kanji_r      = len(re.findall(r'[一-龯]', text)) / max(len(text), 1)
    hira_r       = len(re.findall(r'[ぁ-ん]', text)) / max(len(text), 1)
    pos_w        = sum(text.count(w) for w in POSITIVE_WORDS)
    neg_w        = sum(text.count(w) for w in NEGATIVE_WORDS)
    sentiment    = (pos_w - neg_w) / max(pos_w + neg_w, 1) if (pos_w + neg_w) > 0 else 0.0
    digit_c      = len(re.findall(r'[0-9０-９]', text))
    kata_c       = len(re.findall(r'[ァ-ヶ]', text))
    return [exclamation, ambiguity, symbol, length, digit_r, proper_r,
            avg_sent_len, noun_r, sent_count, kanji_r, hira_r, sentiment, digit_c, kata_c]

# 各特徴量の言い回し（多いとき, 少ないとき）。extract_featuresの並び順と対応
FEATURE_PHRASES = [
    ("感嘆符（！）が多い", "感嘆符（！）が少ない"),
    ("曖昧な表現が多い", "曖昧な表現が少ない"),
    ("記号が多い", "記号が少ない"),
    ("文章が長い", "文章が短い"),
    ("数字の割合が高い", "数字の割合が低い"),
    ("固有名詞が多い", "固有名詞が少ない"),
    ("一文が長い", "一文が短い"),
    ("名詞が多い", "名詞が少ない"),
    ("文の数が多い", "文の数が少ない"),
    ("漢字が多い", "漢字が少ない"),
    ("ひらがなが多い", "ひらがなが少ない"),
    ("肯定的な表現が多い", "否定的な表現が多い"),
    ("数字が多い", "数字が少ない"),
    ("カタカナが多い", "カタカナが少ない"),
]

def make_reasons(features):
    """記事の各特徴量が平均より多い/少ないかで理由を生成"""
    real_mean = real_stats["real_mean"]
    fake_mean = real_stats["fake_mean"]
    values = features[0]

    reasons = []
    for (high_phrase, low_phrase), val, r_m, f_m in zip(FEATURE_PHRASES, values, real_mean, fake_mean):
        # リアルとフェイクの中間を「平均的」の基準にする
        mid = (r_m + f_m) / 2
        gap = f_m - r_m
        if abs(gap) < 1e-9:
            continue  # リアルとフェイクで差がない特徴量は判断材料にしない
        # 基準からどれだけ離れているか（リアル〜フェイクの差を1とした大きさ）
        strength = abs(val - mid) / abs(gap)
        if strength < 0.25:
            continue  # 平均的な範囲は理由にしない
        phrase = high_phrase if val > mid else low_phrase
        reasons.append((strength, phrase))

    # 根拠の強い順に並べる
    reasons.sort(reverse=True)
    return [r[1] for r in reasons]

def predict_reliability(text):
    # 改行・スペース・タブをすべて除去
    text = re.sub(r'\s+', '', text)

    # TF-IDF
    x_tfidf = vectorizer.transform([tokenize_text(text)])

    # 独自特徴量
    features = np.array(extract_features(text)).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # 結合
    x = hstack((x_tfidf, features_scaled))

    # 予測
    proba = model.predict_proba(x)[0]
    fake_score = proba[1] + proba[2]
    reliability = round((1 - fake_score) * 100, 1)

    reasons = make_reasons(features)
    return reliability, reasons

st.title("ニュース信頼度判定")
text = st.text_area("ニュース本文を貼り付けてください")

if st.button("判定する"):
    if not text.strip():
        st.warning("テキストを入力してください")
    else:
        score, reasons = predict_reliability(text)
        st.metric("信頼度", f"{score}%")
        if score >= 70:
            st.success("信頼できる可能性が高い")
        elif score >= 40:
            st.warning("判断が難しい")
        else:
            st.error("フェイクの可能性が高い")

        if reasons:
            st.write("**この記事の特徴:**")
            for r in reasons:
                st.write("・", r)
