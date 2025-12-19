import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== LOAD DATA ====================
df = pd.read_csv("reddit_data_advanced.csv")
print(f"📊 Loaded {len(df)} posts\n")

# ==================== MODELS ====================
vader = SentimentIntensityAnalyzer()

roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# ==================== SCORE COMPUTATION ====================
def compute_scores(text):
    text = str(text)

    # VADER
    vader_score = vader.polarity_scores(text)["compound"]

    # TextBlob
    textblob_score = TextBlob(text).sentiment.polarity

    # RoBERTa
    rob = roberta(text[:512])[0]
    if rob["label"] == "negative":
        roberta_score = -rob["score"]
    elif rob["label"] == "positive":
        roberta_score = rob["score"]
    else:
        roberta_score = 0.0

    return pd.Series([vader_score, textblob_score, roberta_score])

print("🧠 Computing sentiment scores...")
df[["vader_score", "textblob_score", "roberta_score"]] = df["text"].apply(compute_scores)

# ==================== IMPROVED LABELING ====================
def improved_labeling(text, score, subreddit):
    t = str(text).lower()

    very_negative = [
        "suicide", "suicidal", "kill myself", "want to die",
        "hopeless", "worthless", "end my life", "give up"
    ]

    negative = [
        "depressed", "anxious", "sad", "lonely", "overwhelmed",
        "struggling", "exhausted", "stressed"
    ]

    positive = [
        "grateful", "thankful", "better", "improving",
        "hope", "hopeful", "proud", "good day"
    ]

    if any(k in t for k in very_negative):
        return -1
    if any(k in t for k in negative) and not any(k in t for k in positive):
        return -1
    if any(k in t for k in positive):
        return 1

    if subreddit in ["depression", "SuicideWatch"] and score < 10:
        return -1

    if score > 100:
        return 1
    if score < 5:
        return -1

    return 0

print("🏷️ Applying improved labeling...")
df["true_label"] = df.apply(
    lambda r: improved_labeling(r["text"], r["score"], r["subreddit"]),
    axis=1
)

print("\n📊 Label distribution:")
print(df["true_label"].value_counts())

# ==================== ENSEMBLE WEIGHTS ====================
weights = {
    "vader": 0.3,
    "textblob": 0.2,
    "roberta": 0.5
}

df["ensemble_score"] = (
    df["vader_score"] * weights["vader"] +
    df["textblob_score"] * weights["textblob"] +
    df["roberta_score"] * weights["roberta"]
)

df["pred_label"] = df["ensemble_score"].apply(
    lambda x: 1 if x > 0.2 else -1 if x < -0.2 else 0
)

# ==================== EVALUATION ====================
labeled = df[
    (df["true_label"] != 0) &
    (df["pred_label"] != 0)
]

y_true = labeled["true_label"]
y_pred = labeled["pred_label"]

print("\n📊 FINAL EVALUATION")
print(f"Total posts: {len(df)}")
print(f"Evaluated posts: {len(labeled)}")

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy     : {acc:.2%}")
print(f"Macro F1     : {macro_f1:.3f}")
print(f"Weighted F1  : {weighted_f1:.3f}")

print("\n📋 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[-1, 1],
    target_names=["Negative", "Positive"]
))

# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title("Confusion Matrix – Ensemble (VADER + TextBlob + RoBERTa)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("improved_confusion_matrix.png", dpi=300)
plt.show()

# ==================== SAVE ====================
df.to_csv("reddit_advanced_data_improved.csv", index=False)
print("\n💾 Saved reddit_advanced_data_improved.csv")
print("💾 Saved improved_confusion_matrix.png")

print("\n✅ DONE — Ensemble with RoBERTa + Macro & Weighted F1")
