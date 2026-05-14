import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("/Users/yamaguchiharuto-dendai/Downloads/fakenews/fakenews.csv")

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

X = df["context"].apply(extract_features).tolist()
y = df["reliability"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Ridge()
model.fit(X_train, y_train)

joblib.dump(model, "reliability_model.pkl")
print("モデルを保存しました")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"平均誤差: {mae * 100:.1f}%")

for i in range(10):
    reliability = y_pred[i] * 100
    reliability = max(0, min(100, reliability))
    actual = y_test[i] * 100
    print(f"記事{i+1}: 信頼度{reliability:.1f}%  （正解: {actual:.1f}%）")