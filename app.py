
import streamlit as st

from predict import predict
from dataset import load_dataset
from sequential_covering import sequential_covering


# Page configuration
st.set_page_config(
    page_title="Sequential Covering Classifier",
    page_icon="🎾",
    layout="centered"
)


# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #a78bfa;
    }
    .stButton button {
        background-color: #7c3aed;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Project heading
st.title("🎾 Rule-Set Learning for a Toy Classifier")

st.subheader("Sequential Covering Algorithm")

st.write(
    "Predict whether the weather is suitable for playing tennis "
    "using automatically learned IF-THEN rules."
)

st.divider()


# Load dataset and learn rules
df = load_dataset()
learned_rules = sequential_covering(df)


# Weather input section
st.header("Enter Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    outlook = st.selectbox(
        "Outlook",
        ["Sunny", "Overcast", "Rain"]
    )

    humidity = st.selectbox(
        "Humidity",
        ["High", "Normal"]
    )

with col2:
    temp = st.selectbox(
        "Temperature",
        ["Hot", "Mild", "Cool"]
    )

    wind = st.selectbox(
        "Wind",
        ["Weak", "Strong"]
    )


# Prediction
if st.button("Predict"):

    input_data = {
        "outlook": outlook,
        "temp": temp,
        "humidity": humidity,
        "wind": wind
    }

    result = predict(input_data)

    st.subheader("Classification Result")

    if result["prediction"] == "Yes":
        st.success("Predicted Class: YES")
    else:
        st.error("Predicted Class: NO")

    st.info(result["rule"])


st.divider()


# Display learned rules
st.header("Learned IF-THEN Rules")

for i, item in enumerate(learned_rules, start=1):

    conditions = " AND ".join(
        f"{feature} = {value}"
        for feature, value in item["rule"].items()
    )

    with st.expander(f"Rule {i}", expanded=True):

        st.write(
            f"**IF** {conditions} **THEN** play = Yes"
        )

        st.write(
            f"Records covered: {len(item['covered_records'])}"
        )

        st.write(
            f"Rule quality: {item['quality']:.2%}"
        )


# Overall results
st.divider()

st.header("Experimental Results")

total_positive = len(df[df["play"] == "Yes"])

covered_positive = sum(
    item["true_positives"] for item in learned_rules
)

st.metric(
    "Total Rules Generated",
    len(learned_rules)
)

st.metric(
    "Positive Examples Covered",
    f"{covered_positive}/{total_positive}"
)

st.metric(
    "Positive Coverage",
    f"{covered_positive / total_positive * 100:.2f}%"
    if total_positive else "0%"
)


# Footer
st.divider()

st.caption("Project: Rule-Set Learning for a Toy Classifier")
st.caption("Method: Sequential Covering")
st.caption("Dataset: Kaggle Play Tennis Dataset")
