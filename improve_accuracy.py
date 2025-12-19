import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv('reddit_data.csv')
print(f"📊 Loaded {len(df)} posts\n")

# ----------------------------
# Initialize models
# ----------------------------
vader = SentimentIntensityAnalyzer()

# ==================== BETTER LABELING ====================
def improved_labeling(text, score, subreddit):
    """
    Better labeling strategy combining multiple signals
    """
    text_lower = str(text).lower()
    
    very_negative = [
        'suicide', 'suicidal', 'kill myself', 'want to die', 
        'hopeless', 'worthless', "can't go on", 'end my life',
        'better off dead', 'no point', 'give up'
    ]
    
    negative = [
        'depressed', 'anxious', 'sad', 'lonely', 'overwhelmed',
        'struggling', 'difficult', 'hard time', "can't cope",
        'exhausted', 'tired', 'stressed', 'worried', 'afraid'
    ]
    
    positive = [
        'grateful', 'thankful', 'better', 'improving', 'progress',
        'hope', 'hopeful', 'proud', 'accomplished', 'good day',
        'helpful', 'working', 'success'
    ]
    
    very_positive = [
        'amazing', 'wonderful', 'fantastic', 'great', 'excellent',
        'best day', 'so happy', 'celebrating', 'milestone'
    ]
    
    has_very_negative = any(kw in text_lower for kw in very_negative)
    has_negative = any(kw in text_lower for kw in negative)
    has_positive = any(kw in text_lower for kw in positive)
    has_very_positive = any(kw in text_lower for kw in very_positive)
    
    if has_very_negative:
        return -1
    elif has_very_positive and not has_negative:
        return 1
    elif has_positive and not has_negative:
        return 1
    elif has_negative and not has_positive:
        return -1
    
    if subreddit in ['depression', 'SuicideWatch']:
        if score < 10:
            return -1
    
    if score > 100:
        return 1
    elif score < 5:
        return -1
    
    return 0  # Neutral

print("🏷️  Applying improved labeling...")
df['true_label'] = df.apply(
    lambda row: improved_labeling(row['text'], row['score'], row['subreddit']),
    axis=1
)

print(f"\n📊 Label distribution:")
print(df['true_label'].value_counts())

# ==================== IMPROVE MODEL WEIGHTS ====================
print("\n🔧 Testing different ensemble weights...")

# Ensure your df has vader_score and textblob_score columns
df['vader_score'] = df['text'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
df['textblob_score'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

weight_configs = [
    {'vader': 1.0, 'textblob': 0.0, 'name': 'VADER only'},
    {'vader': 0.0, 'textblob': 1.0, 'name': 'TextBlob only'},
    {'vader': 0.6, 'textblob': 0.4, 'name': 'Current (60/40)'},
    {'vader': 0.7, 'textblob': 0.3, 'name': 'VADER Heavy (70/30)'},
    {'vader': 0.8, 'textblob': 0.2, 'name': 'VADER Very Heavy (80/20)'},
    {'vader': 0.5, 'textblob': 0.5, 'name': 'Equal (50/50)'},
]

results = []

for config in weight_configs:
    df['ensemble_test'] = (
        df['vader_score'] * config['vader'] +
        df['textblob_score'] * config['textblob']
    )
    
    df['pred_label'] = df['ensemble_test'].apply(
        lambda x: 1 if x > 0.2 else -1 if x < -0.2 else 0
    )
    
    # Only consider posts with true labels -1 or 1 AND exclude neutral predictions
    labeled = df[(df['true_label'] != 0) & (df['pred_label'] != 0)]
    if len(labeled) > 0:
        accuracy = accuracy_score(labeled['true_label'], labeled['pred_label'])
        results.append({
            'config': config['name'],
            'vader_weight': config['vader'],
            'textblob_weight': config['textblob'],
            'accuracy': accuracy
        })
        print(f"   {config['name']}: {accuracy:.1%}")

# Best configuration
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['accuracy'].idxmax()]

print(f"\n🏆 BEST CONFIGURATION:")
print(f"   {best['config']}")
print(f"   Accuracy: {best['accuracy']:.1%}")
print(f"   Weights: VADER={best['vader_weight']}, TextBlob={best['textblob_weight']}")

# Apply best weights
df['ensemble_score'] = (
    df['vader_score'] * best['vader_weight'] +
    df['textblob_score'] * best['textblob_weight']
)

df['pred_label'] = df['ensemble_score'].apply(
    lambda x: 1 if x > 0.2 else -1 if x < -0.2 else 0
)

# ==================== DETAILED EVALUATION ====================
# Exclude neutral predictions for evaluation
labeled = df[(df['true_label'] != 0) & (df['pred_label'] != 0)]

print(f"\n📊 FINAL EVALUATION:")
print(f"   Total posts: {len(df)}")
print(f"   Labeled posts: {len(labeled)}")
print(f"   Final accuracy: {accuracy_score(labeled['true_label'], labeled['pred_label']):.1%}")

print(f"\n📋 Classification Report:")
print(classification_report(
    labeled['true_label'], 
    labeled['pred_label'],
    labels=[-1, 1],
    target_names=['Negative', 'Positive']
))

# Confusion matrix
cm = confusion_matrix(
    labeled['true_label'], 
    labeled['pred_label'],
    labels=[-1, 1]
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix - {best["config"]}\nAccuracy: {best["accuracy"]:.1%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png', dpi=300)
print(f"\n💾 Saved confusion matrix to 'improved_confusion_matrix.png'")

# Save improved data
df.to_csv('reddit_data_improved.csv', index=False)
print(f"💾 Saved improved dataset to 'reddit_data_improved.csv'")

# ==================== ANALYZE ERRORS ====================
print(f"\n🔍 Analyzing errors...")

errors = labeled[labeled['true_label'] != labeled['pred_label']]
print(f"   Total errors: {len(errors)}")

if len(errors) > 0:
    print(f"\n❌ Sample errors:")
    for idx, row in errors.head(5).iterrows():
        print(f"\n   Text: {row['text'][:100]}...")
        print(f"   True: {row['true_label']}, Predicted: {row['pred_label']}")
        print(f"   Sentiment: {row['ensemble_score']:.3f}")

print(f"\n✅ Analysis complete!")
print(f"\n🎯 To use improved model:")
print(f"   1. Use weights: VADER={best['vader_weight']}, TextBlob={best['textblob_weight']}")
print(f"   2. Copy 'reddit_data_improved.csv' to 'reddit_data.csv'")
print(f"   3. Restart your dashboard")
