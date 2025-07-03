
# 🧠 Mind the Gap: Analyzing Global Mental Health Conversations Using AI

An AI-powered NLP pipeline that tracks and analyzes emotional trends in mental health discussions across the globe.*

## 📌 Overview

**Mind the Gap** is a data-driven research project that leverages **Natural Language Processing (NLP)** to explore global mental health conversations on platforms like Reddit. The project uses **AI models** for **sentiment analysis** and **topic modeling** to uncover emotional patterns, recurring themes, and regional trends over time.

It highlights how AI can support mental health awareness by providing meaningful insights into public discourse, making it valuable for researchers, policymakers, and educators.

---

## 🚀 Key Features

- 📊 **Sentiment Analysis** on thousands of social media posts  
- 🌍 **Global Trends Visualization** over time  
- 🧠 **Topic Modeling** to identify recurring mental health themes  
- ⚡️ **Streamlit Dashboard** for real-time exploration  
- 🧪 Research-oriented design with ethics and scalability in mind

---

## 🧱 Project Structure

```
mind-the-gap/
│
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned, labeled sentiment data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   └── 03_topic_modeling.ipynb
│
├── app/
│   └── streamlit_app.py   # Interactive visualization
│
├── README.md
└── requirements.txt
```

---

## 📂 Dataset

We used the [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/infamouscoder/reddit-mental-health-dataset), which includes:
- Post titles and content
- Timestamps and subreddit metadata
- Conversations around anxiety, depression, and self-care

---

## 🛠️ Tech Stack

| Tool/Library        | Purpose                      |
|---------------------|------------------------------|
| **Python**          | Core scripting language       |
| **Pandas**          | Data wrangling & cleaning     |
| **TextBlob / VADER**| Sentiment Analysis (NLP)      |
| **Scikit-Learn**    | Topic modeling (LDA)          |
| **Plotly / Streamlit** | Dashboard & visualization |

---

## 📈 Sample Output

- 📉 **Sentiment Trends Over Time**
- 🧩 **Emotion Distribution (Positive/Neutral/Negative)**
- 🔍 **Top Discussion Topics** like "therapy", "isolation", "stress", "recovery"

---

## 📘 Research Statement

> Mental health awareness is growing globally, yet the emotional tone and underlying themes of online conversations vary significantly across cultures and time. This project applies interpretable AI to detect patterns in global sentiment and identify actionable insights for improving mental health support and education.

---

## 💡 Future Scope

- 🎭 Deep Learning-based Emotion Detection (BERT, RoBERTa)
- 🌐 Multilingual NLP for broader demographic coverage
- 🧬 Resource Recommender for users based on text inputs
- 📡 Real-time social media monitoring via APIs

---



Made by Krisha Oswal 


## 🌟 Show Your Support

If you found this project helpful or inspiring:
- ⭐️ Star this repository
- 🧠 Share it with mental health or research communities
- 📝 Feel free to contribute or cite us!
