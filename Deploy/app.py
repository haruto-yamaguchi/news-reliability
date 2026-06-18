import joblib
import fugashi
import re
import numpy as np
from scipy.sparse import hstack

# モデルの読み込み
model = joblib.load("Deploy/models/kadai003_model.pkl")
vectorizer = joblib.load("Deploy/models/kadai003_vectorizer.pkl")
scaler = joblib.load("Deploy/models/kadai003_scaler.pkl")
real_stats = joblib.load("Deploy/models/kadai003_real_stats.pkl")

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

# 各特徴量のわかりやすい名前
FEATURE_LABELS = [
    "感嘆符（！）",
    "曖昧な表現",
    "記号",
    "文章の長さ",
    "数字の割合",
    "固有名詞の割合",
    "一文の長さ",
    "名詞の割合",
    "文の数",
    "漢字の割合",
    "ひらがなの割合",
    "感情の偏り",
    "数字の数",
    "カタカナの数",
]

def make_reasons(features):
    """リアル記事とフェイク記事の平均、どちら寄りかで理由を生成"""
    real_mean = real_stats["real_mean"]
    fake_mean = real_stats["fake_mean"]
    values = features[0]

    reasons = []
    for label, val, r_m, f_m in zip(FEATURE_LABELS, values, real_mean, fake_mean):
        gap = f_m - r_m
        # リアルとフェイクの平均がほぼ同じ特徴量は判断材料にならないので除外
        if abs(gap) < 1e-9:
            continue
        # この記事の値が、リアル(0.0)〜フェイク(1.0)のどの位置にあるか
        pos = (val - r_m) / gap
        # 0.5を境にどちら寄りかを判定。0.5から離れているほど強い根拠
        strength = abs(pos - 0.5)
        if strength < 0.25:
            continue  # 中間付近は理由にしない
        if pos > 0.5:
            side = "フェイク記事に近い"
        else:
            side = "リアル記事に近い"
        reasons.append((strength, f"{label}が{side}"))

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

if __name__ == "__main__":
    import tkinter as tk

    def on_judge():
        text = text_box.get("1.0", tk.END).strip()
        if not text:
            result_label.config(text="テキストを入力してください", fg="gray")
            reason_label.config(text="")
            return
        score, reasons = predict_reliability(text)
        if score >= 70:
            judgment = "信頼できる可能性が高い"
            color = "green"
        elif score >= 40:
            judgment = "判断が難しい"
            color = "orange"
        else:
            judgment = "フェイクの可能性が高い"
            color = "red"
        result_label.config(text=f"信頼度: {score}%　{judgment}", fg=color)

        if reasons:
            reason_text = "この記事の特徴:\n" + "\n".join(f"・{r}" for r in reasons)
        else:
            reason_text = "この記事の特徴: 平均的な文章です"
        reason_label.config(text=reason_text)

import streamkit as st

st.title("ニュース信頼度判定")
text = st.text_area("ニュース本文を貼り付けてください")

if st.button("判定する"):
    score, reasons = predict_reliability(text)
    st.write(f"信頼度: {score}%")
    for r in reasons:
        st.write("・", r)
