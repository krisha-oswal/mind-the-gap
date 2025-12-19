import praw
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from tqdm import tqdm
import time

# Load credentials
load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

def collect_subreddit_data(subreddit_name, limit=500):
    """
    Collect posts using official Reddit API with proper authentication
    """
    print(f"\n📡 Collecting from r/{subreddit_name}")
    
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    for submission in tqdm(subreddit.hot(limit=limit), total=limit, desc="Fetching"):
        posts_data.append({
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'author': str(submission.author),
            'url': submission.url,
            'subreddit': subreddit_name,
            'upvote_ratio': submission.upvote_ratio,
            'flair': submission.link_flair_text
        })
    
    return pd.DataFrame(posts_data)

# Collect from multiple subreddits
subreddits = ['mentalhealth', 'depression', 'anxiety', 'therapy', 'selfimprovement']

all_data = []
for sub in subreddits:
    df = collect_subreddit_data(sub, limit=300)
    all_data.append(df)
    time.sleep(2)  # Rate limiting

# Combine
dataset = pd.concat(all_data, ignore_index=True)
dataset.to_csv('reddit_data_advanced.csv', index=False)

print(f"\n✅ Collected {len(dataset)} posts")
print(f"\n📊 Distribution:")
print(dataset['subreddit'].value_counts())