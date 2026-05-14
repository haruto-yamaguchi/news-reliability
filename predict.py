import joblib

model = joblib.load("reliability_model.pkl")

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

print("ニュース本文を入力 : ")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

text = "\n".join(lines)

features = extract_features(text)
pred = model.predict([features])[0]
reliability = max(0, min(100, pred * 100))

print(f"\n信頼度: {reliability:.1f}%")