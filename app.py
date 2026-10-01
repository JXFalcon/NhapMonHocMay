import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Naive Bayes - Bai 1 & 2", layout="wide")
st.title("Dự đoán Naive Bayes — Bài 1 & Bài 2")

# ---------- Data ----------
DATA1 = [
    ("Sunny", "Hot", "High", "False", "No"),
    ("Sunny", "Hot", "High", "True", "No"),
    ("Overcast", "Hot", "High", "False", "Yes"),
    ("Rain", "Mild", "High", "False", "Yes"),
    ("Rain", "Cool", "Normal", "False", "Yes"),
    ("Rain", "Cool", "Normal", "True", "No"),
    ("Overcast", "Cool", "Normal", "True", "Yes"),
    ("Sunny", "Mild", "High", "False", "No"),
    ("Sunny", "Cool", "Normal", "False", "Yes"),
    ("Rain", "Mild", "Normal", "False", "Yes"),
]
FEATURES1 = ["Weather", "Temp", "Humidity", "Windy"]

DATA2 = [
    ("youth", "high", "no", "fair", "no"),
    ("youth", "high", "no", "excellent", "no"),
    ("middle", "high", "no", "fair", "yes"),
    ("senior", "medium", "no", "fair", "yes"),
    ("senior", "low", "yes", "fair", "yes"),
    ("senior", "low", "yes", "excellent", "no"),
    ("middle", "low", "yes", "excellent", "yes"),
    ("youth", "medium", "no", "fair", "yes"),
    ("youth", "low", "yes", "fair", "yes"),
    ("senior", "medium", "yes", "fair", "yes"),
    ("middle", "medium", "no", "excellent", "yes"),
    ("youth", "medium", "yes", "excellent", "yes"),
    ("middle", "high", "yes", "fair", "yes"),
    ("senior", "medium", "no", "excellent", "no"),
    ("youth", "low", "no", "fair", "no"),
]
FEATURES2 = ["Age", "Income", "Student", "Credit"]


def train(data):
    total = len(data)
    counts = Counter(r[4] for r in data)
    prior = {c: counts[c] / total for c in counts}
    likelihood = {c: {} for c in counts}
    for c in counts:
        rows = [r for r in data if r[4] == c]
        n = len(rows)
        for i in range(4):
            cc = Counter(r[i] for r in rows)
            likelihood[c][i] = {v: cc[v] / n for v in cc}
    return prior, likelihood, counts, total


def predict(x, prior, likelihood):
    scores = {}
    for c in prior:
        p = prior[c]
        for i, v in enumerate(x):
            p *= likelihood[c][i].get(v, 0.0)
        scores[c] = p
    s = sum(scores.values())
    proba = {c: (scores[c] / s if s > 0 else 0.0) for c in scores}
    return scores, proba


def render_bai(data, features, label_name, defaults, options):
    df = pd.DataFrame(data, columns=features + [label_name])
    st.subheader("Dữ liệu huấn luyện")
    st.dataframe(df, use_container_width=True)

    prior, likelihood, counts, total = train(data)

    st.subheader("Nhập điều kiện dự đoán")
    cols = st.columns(4)
    x = []
    for i, f in enumerate(features):
        with cols[i]:
            x.append(st.selectbox(f, options[i], index=options[i].index(defaults[i]), key=f"{label_name}-{f}"))
    x = tuple(x)

    if st.button(f"Dự đoán {label_name}", key=f"btn-{label_name}"):
        scores, proba = predict(x, prior, likelihood)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Priors** (tổng {total} mẫu):")
            for c, p in prior.items():
                st.write(f"P({c}) = {counts[c]}/{total} = {p:.4f}")
            st.write("**Chi tiết Score:**")
            for c in prior:
                parts = [f"P({c})={prior[c]:.4f}"]
                for i, v in enumerate(x):
                    parts.append(f"P({features[i]}={v}|{c})={likelihood[c][i].get(v, 0.0):.4f}")
                st.write(f"Score({c}) = {' × '.join(parts)} = **{scores[c]:.6f}**")
        with c2:
            st.write("**Xác suất sau chuẩn hóa:**")
            for c in prior:
                st.write(f"P({c}|X) = {proba[c]*100:.2f}%")
            st.bar_chart(pd.DataFrame({"Xác suất": proba}))

        ket = max(proba, key=proba.get)
        st.success(f"KẾT LUẬN: {label_name} = {ket.upper()}  (X={x})")


tab1, tab2 = st.tabs(["Bài 1: Play Tennis", "Bài 2: Buy Laptop"])

with tab1:
    st.header("Bài 1: X = (Sunny, Cool, High, True)")
    render_bai(DATA1, FEATURES1, "Play",
               ("Sunny", "Cool", "High", "True"),
               [["Sunny", "Overcast", "Rain"], ["Hot", "Mild", "Cool"],
                ["High", "Normal"], ["False", "True"]])

with tab2:
    st.header("Bài 2: X = (youth, medium, yes, fair)")
    render_bai(DATA2, FEATURES2, "BuyLaptop",
               ("youth", "medium", "yes", "fair"),
               [["youth", "middle", "senior"], ["high", "medium", "low"],
                ["yes", "no"], ["fair", "excellent"]])
