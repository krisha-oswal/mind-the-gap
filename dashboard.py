import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Mind the Gap",
    page_icon="🧠",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    if os.path.exists('reddit_data.csv'):
        df = pd.read_csv('reddit_data.csv')
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        return df
    return None

# ==================== CREATE DEMO DATA ====================
def create_demo_data():
    import random
    from datetime import timedelta
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    
    vader = SentimentIntensityAnalyzer()
    
    texts = [
        "I'm feeling really anxious about work today",
        "Had a great therapy session! Feeling hopeful about the future",
        "Another sleepless night. This is getting exhausting",
        "Went for a walk and felt so peaceful. Nature really helps",
        "Feeling completely overwhelmed with everything right now",
        "Proud of myself for reaching out to a friend today",
        "Can't shake this feeling of sadness. It's been weeks",
        "Had an amazing day with family. Feeling grateful",
        "Work stress is really getting to me. Need better balance",
        "Meditation is really helping me feel more calm",
        "Feeling lonely even when surrounded by people",
        "Small wins today - got out of bed and went to the gym!",
        "Anxiety is through the roof. Panic attacks are back",
        "Finally found a medication that works. Things are improving!",
        "Feeling hopeless. Don't know how much longer I can do this",
        "Support group was incredible today. Not alone in this journey",
        "Exhausted from pretending everything is fine",
        "Celebrating 3 months of consistent self-care!",
        "Financial stress is affecting my mental health badly",
        "Therapy gave me tools to manage my anxiety better"
    ]
    
    subreddits = ['mentalhealth', 'anxiety', 'depression', 'therapy']
    
    posts = []
    for i in range(200):
        text = random.choice(texts)
        v = vader.polarity_scores(text)['compound']
        t = TextBlob(text).sentiment.polarity
        ensemble = v * 0.7 + t * 0.3
        
        crisis_words = ['suicide', 'hopeless', 'can\'t go on', 'end my life']
        is_crisis = any(w in text.lower() for w in crisis_words)
        
        if is_crisis and ensemble < -0.7:
            crisis_level = 'CRITICAL'
        elif is_crisis or ensemble < -0.7:
            crisis_level = 'HIGH'
        elif ensemble < -0.4:
            crisis_level = 'MODERATE'
        else:
            crisis_level = 'LOW'
        
        posts.append({
            'id': f'post_{i}',
            'title': text[:50] + '...',
            'text': text,
            'score': random.randint(1, 500),
            'num_comments': random.randint(0, 100),
            'created_utc': datetime.now() - timedelta(days=random.randint(0, 30)),
            'author': f'user_{random.randint(1, 50)}',
            'subreddit': random.choice(subreddits),
            'vader_score': v,
            'textblob_score': t,
            'ensemble_score': ensemble,
            'crisis_level': crisis_level
        })
    
    df = pd.DataFrame(posts)
    df.to_csv('reddit_data.csv', index=False)
    return df

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; font-size: 3.5em; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
        🧠 MIND THE GAP
    </h1>
    <p style='text-align: center; font-size: 1.3em; color: white;'>
        Real-time Reddit Sentiment Analysis...
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
    
    # Load or create data
    df = load_data()
    
    if df is None:
        st.sidebar.error("❌ No data found")
        if st.sidebar.button("🎲 Generate Demo Data", type="primary"):
            with st.spinner("Creating demo data..."):
                df = create_demo_data()
                st.success("✅ Created 200 demo posts!")
                st.rerun()
        st.info("👈 Click 'Generate Demo Data' in the sidebar to start")
        return
    
    st.sidebar.success(f"✅ {len(df)} posts loaded")
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    # Subreddit filter
    all_subreddits = df['subreddit'].unique().tolist()
    selected_subreddits = st.sidebar.multiselect(
        "Subreddits",
        options=all_subreddits,
        default=all_subreddits
    )
    
    # Apply filters
    filtered_df = df[df['subreddit'].isin(selected_subreddits)]
    
    # Date range
    if 'created_utc' in filtered_df.columns:
        min_date = filtered_df['created_utc'].min().date()
        max_date = filtered_df['created_utc'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['created_utc'].dt.date >= date_range[0]) &
                (filtered_df['created_utc'].dt.date <= date_range[1])
            ]
    
    st.sidebar.caption(f"🕐 Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # ==================== METRICS ====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Posts",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        avg_sentiment = filtered_df['ensemble_score'].mean()
        st.metric(
            "💭 Avg Sentiment",
            f"{avg_sentiment:.3f}",
            delta=f"{'↑' if avg_sentiment > 0 else '↓'}"
        )
    
    with col3:
        crisis = len(filtered_df[filtered_df['crisis_level'].isin(['HIGH', 'CRITICAL'])])
        st.metric(
            "🚨 Crisis Posts",
            crisis,
            delta=f"{(crisis/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "0%",
            delta_color="inverse"
        )
    
    with col4:
        positive = len(filtered_df[filtered_df['ensemble_score'] > 0.2])
        st.metric(
            "😊 Positive Posts",
            positive,
            delta=f"{(positive/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "0%"
        )
    
    # Crisis Alert
    if crisis > 0:
        st.error(f"""
        ⚠️ **ALERT**: {crisis} posts flagged as HIGH or CRITICAL risk
        
        **Crisis Resources:**
        - 🇺🇸 National Suicide Prevention Lifeline: **988**
        - 💬 Crisis Text Line: Text **HELLO** to **741741**
        - 🌍 International: findahelpline.com
        """)
    
    st.markdown("---")
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trends", 
        "📊 Distribution", 
        "🔍 Analysis",
        "📝 Posts"
    ])
    
    with tab1:
        st.subheader("Sentiment Trends Over Time")
        
        if 'created_utc' in filtered_df.columns:
            # Daily aggregation
            daily = filtered_df.set_index('created_utc').resample('D').agg({
                'ensemble_score': 'mean',
                'id': 'count'
            }).reset_index()
            daily.columns = ['date', 'sentiment', 'count']
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Daily Average Sentiment", "Post Volume"),
                vertical_spacing=0.1
            )
            
            # Sentiment line
            fig.add_trace(
                go.Scatter(
                    x=daily['date'],
                    y=daily['sentiment'],
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='#00d4ff', width=3),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1)
            
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=daily['date'],
                    y=daily['count'],
                    name='Posts',
                    marker_color='#a855f7'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date information available")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            
            fig = px.histogram(
                filtered_df,
                x='ensemble_score',
                nbins=40,
                title="",
                labels={'ensemble_score': 'Sentiment Score'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Crisis Levels")
            
            crisis_counts = filtered_df['crisis_level'].value_counts()
            colors_map = {
                'LOW': '#10b981',
                'MODERATE': '#f59e0b',
                'HIGH': '#ef4444',
                'CRITICAL': '#dc2626'
            }
            colors = [colors_map.get(level, '#6b7280') for level in crisis_counts.index]
            
            fig = go.Figure(data=[go.Pie(
                labels=crisis_counts.index,
                values=crisis_counts.values,
                marker=dict(colors=colors),
                hole=0.4
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Subreddit Comparison")
        
        subreddit_stats = filtered_df.groupby('subreddit').agg({
            'ensemble_score': 'mean',
            'id': 'count'
        }).reset_index()
        subreddit_stats.columns = ['subreddit', 'avg_sentiment', 'post_count']
        
        fig = px.bar(
            subreddit_stats,
            x='subreddit',
            y='avg_sentiment',
            color='post_count',
            title="Average Sentiment by Subreddit",
            labels={'avg_sentiment': 'Avg Sentiment', 'post_count': 'Posts'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_data = pd.DataFrame({
                'Model': ['VADER', 'TextBlob', 'Ensemble'],
                'Avg Score': [
                    filtered_df['vader_score'].mean(),
                    filtered_df['textblob_score'].mean(),
                    filtered_df['ensemble_score'].mean()
                ]
            })
            
            fig = px.bar(
                model_data,
                x='Model',
                y='Avg Score',
                title="Model Score Comparison",
                color='Model'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation
            corr_data = filtered_df[['vader_score', 'textblob_score', 'ensemble_score']].corr()
            
            fig = px.imshow(
                corr_data,
                title="Model Correlation",
                color_continuous_scale='RdBu',
                aspect='auto',
                text_auto='.2f'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Recent Posts")
        
        # Sort by date
        recent = filtered_df.sort_values('created_utc', ascending=False).head(20)
        
        for idx, row in recent.iterrows():
            sentiment_emoji = "🟢" if row['ensemble_score'] > 0.2 else "🔴" if row['ensemble_score'] < -0.2 else "🟡"
            crisis_emoji = "🚨" if row['crisis_level'] in ['HIGH', 'CRITICAL'] else ""
            
            with st.expander(f"{sentiment_emoji} {crisis_emoji} {row['title']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Subreddit:** r/{row['subreddit']}")
                    if 'created_utc' in row:
                        st.write(f"**Posted:** {row['created_utc'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Text:** {row['text'][:300]}...")
                
                with col2:
                    st.metric("Sentiment", f"{row['ensemble_score']:.3f}")
                    st.metric("Upvotes", row['score'])
                
                with col3:
                    st.metric("Crisis", row['crisis_level'])
                    st.metric("Comments", row['num_comments'])
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()