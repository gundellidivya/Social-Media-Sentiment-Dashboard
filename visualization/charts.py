import streamlit as st

def plot_sentiment_trend(df):
    trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(trend)

def plot_sentiment_distribution(df):
    st.bar_chart(df['sentiment'].value_counts())

def display_word_cloud(df):
    from wordcloud import WordCloud
    text = " ".join(df['cleaned_text'].tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wc.to_array())
