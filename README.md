# MIND THE GAP: Reddit Mental Health Sentiment Analysis

**Ensemble Learning, Transformer Models, and Explainable AI**

## Overview

This project focuses on building a sentiment analysis pipeline for Reddit posts related to mental health. Online mental-health discussions are often emotionally complex, informal, and noisy, which makes automated sentiment detection particularly challenging.

To address this, the project combines multiple sentiment analysis approaches—lexicon-based, statistical, and transformer-based models—into a single ensemble system. The goal is not only to improve prediction quality, but also to understand *why* the model makes certain predictions using explainability techniques.

---

## Objectives

The main objectives of this project are:

* To analyze sentiment in mental health–related Reddit discussions
* To reduce reliance on a single sentiment model by using an ensemble approach
* To handle weakly labeled and imbalanced data realistically
* To evaluate performance using appropriate metrics beyond accuracy
* To provide interpretability for model outputs

---

## Project Structure

```
reddit/
│
├── improve_accuracy.py          # Main analysis and evaluation script
├── reddit_advanced_data.csv     # Full processed dataset
├── reddit_data_improved.csv     # Dataset with predicted labels
│
├── improved_confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset consists of posts collected from mental health–related subreddits.
The primary file used in this project is `reddit_advanced_data.csv`.

### Important Columns

| Column           | Description                        |
| ---------------- | ---------------------------------- |
| `processed_text` | Cleaned and preprocessed post text |
| `subreddit`      | Source subreddit                   |
| `score`          | Reddit post engagement score       |
| `vader_score`    | Sentiment score from VADER         |
| `textblob_score` | Sentiment score from TextBlob      |
| `roberta_score`  | Sentiment score from RoBERTa       |
| `ensemble_score` | Weighted sentiment score           |
| `true_label`     | Weakly supervised sentiment label  |
| `pred_label`     | Final predicted sentiment          |

---

## Models Used

### VADER

VADER is a rule-based sentiment analyzer designed for social-media text. It performs well on short and informal content and provides fast, interpretable results.

### TextBlob

TextBlob uses statistical techniques to compute polarity scores. It is simple to use and often performs well on grammatically clean sentences.

### RoBERTa

RoBERTa is a transformer-based model that captures contextual meaning in text.
This project uses the pretrained model:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

---

## Ensemble Method

Instead of relying on a single model, sentiment predictions are combined using a weighted ensemble approach:

```
ensemble_score =
    w1 × vader_score +
    w2 × textblob_score +
    w3 × roberta_score
```

Different weight combinations are evaluated, and the configuration with the best performance on labeled data is selected.

---

## Labeling Strategy

Because the dataset does not contain human-annotated labels, a weak supervision strategy is used. Labels are inferred using:

* Presence of strong mental-health keywords
* Reddit engagement score
* Subreddit context
* Sentiment polarity thresholds

The final labels are:

* `-1` for negative sentiment
* `0` for neutral sentiment
* `1` for positive sentiment

This approach allows experimentation while acknowledging that the labels are approximate.

---

## Evaluation Metrics

Accuracy alone is not sufficient due to class imbalance. Therefore, the following metrics are used:

* Accuracy
* Macro F1-score (treats all classes equally)
* Weighted F1-score (accounts for class imbalance)
* Confusion matrix

In this context, an accuracy around 69–71% is considered reasonable given the noisy text and weak labeling.

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain model behavior.
Only numerical sentiment features are passed to SHAP to ensure meaningful and stable explanations.

This helps identify which sentiment scores contribute most to final predictions.

---

## How to Run the Project

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Analysis

```bash
python improve_accuracy.py
```

---

## Outputs

Running the script produces:

* `reddit_data_improved.csv` containing predicted labels
* A confusion matrix saved as `improved_confusion_matrix.png`
* Console output showing accuracy, macro F1, weighted F1, and best ensemble weights

---

## Challenges Faced

During development, several issues were encountered:

* Missing dependencies such as `vaderSentiment`
* Incorrect dataset loading leading to partial data usage
* Evaluation errors caused by mismatched class labels
* Misleading accuracy due to filtered samples
* SHAP dimensionality issues with text embeddings

Each issue was resolved through dependency fixes, explicit label handling, proper metric configuration, and careful validation of intermediate outputs.

---

## Limitations

* Labels are weakly supervised and not human-annotated
* Neutral sentiment is under-represented
* Sarcasm and implicit emotional cues are difficult to capture
* Reddit language is informal and highly variable

---

## Future Work

* Fine-tuning transformer models on mental-health-specific text
* Adding emotion-level classification (e.g., anxiety, sadness, hope)
* Creating a small human-annotated validation set
* Deploying the system as a real-time analysis dashboard



