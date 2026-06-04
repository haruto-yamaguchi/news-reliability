import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import fugashi

# データ読み込み
df = pd.read_csv("/Users/yamaguchiharuto-dendai/Downloads/fakenews/fakenews.csv")

# 目的変数の作成（信頼度 0〜1）
def make_label(row):
    if row["isfake"] == 0:
        return 1.0
    elif row["isfake"] == 2:
        return 0.0
    else:
        total = row["nchar_real"] + row["nchar_fake"]
        if total == 0:
            return 0.5
        return row["nchar_real"] / total

df["reliability"] = df.apply(make_label, axis=1)

# ==============================
# 分かち書き（日本語の単語分割）
# ==============================
tagger = fugashi.Tagger()

def tokenize(text):
    return " ".join([word.surface for word in tagger(text)])

print("分かち書き処理中...（少し時間がかかります）")
df["tokenized"] = df["context"].apply(tokenize)
print("完了")

# ==============================
# 手動特徴量の抽出
# ==============================
def extract_features(text):
    q_mark = text.count("?") + text.count("？")
    quote = text.count("「")
    source = (text.count("によれば") +
              text.count("によると"))
    ambiguous = (text.count("だろう") +
                 text.count("かも") +
                 text.count("と思われ"))
    conjunction = (text.count("そして") +
                   text.count("ただし"))
    return [q_mark, quote, source, ambiguous, conjunction]

# ==============================
# 特徴量の結合
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df["tokenized"])

X_manual = np.array(df["context"].apply(extract_features).tolist())

X_combined = hstack([X_tfidf, X_manual])

y = df["reliability"].tolist()

# ==============================
# データ分割
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 学習
# ==============================
model = Ridge()
model.fit(X_train, y_train)

# ==============================
# モデルの保存
# ==============================
joblib.dump(model, "reliability_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("モデルを保存しました")

# ==============================
# 評価
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"平均誤差: {mae * 100:.1f}%")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

for i in range(20):
    reliability = y_pred[i] * 100
    reliability = max(0, min(100, reliability))
    actual = y_test[i] * 100
    print(f"記事{i+1}: 信頼度{reliability:.1f}%  （正解: {actual:.1f}%）")

    import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 1], [0, 1], "r--", label="理想")  # 対角線
plt.xlabel("正解の信頼度")
plt.ylabel("予測した信頼度")
plt.title("予測値 vs 正解値")
plt.legend()
plt.tight_layout()
plt.savefig("scatter.png")
plt.show()

errors = np.array(y_pred) - np.array(y_test)

plt.figure(figsize=(7, 4))
plt.hist(errors, bins=30, edgecolor="black")
plt.axvline(0, color="r", linestyle="--", label="誤差ゼロ")
plt.xlabel("予測誤差（予測 - 正解）")
plt.ylabel("件数")
plt.title("予測誤差の分布")
plt.legend()
plt.tight_layout()
plt.savefig("error_hist.png")
plt.show()
