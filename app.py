import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

st.set_page_config(page_title="Healthcare Sentiment Dashboard", layout="wide")

df = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

df["Cleaned_Tweet"] = df["OriginalTweet"].apply(clean_text)
df["Sentiment"] = df["Sentiment"].replace({"Extremely Negative": "Negative", "Extremely Positive": "Positive"})
df["TweetAt"] = pd.to_datetime(df["TweetAt"], errors='coerce')

st.title("🩺Healthcare Social Media Sentiment Dashboard")

st.sidebar.title("🔍 Filters")
sentiments = st.sidebar.multiselect("Choose Sentiment", df["Sentiment"].unique(), default=df["Sentiment"].unique())
min_date = df["TweetAt"].min()
max_date = df["TweetAt"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

filtered_df = df[
    (df["Sentiment"].isin(sentiments)) &
    (df["TweetAt"] >= pd.to_datetime(date_range[0])) &
    (df["TweetAt"] <= pd.to_datetime(date_range[1]))
]

st.markdown("## 📈 Sentiment Trend Over Time")
trend_data = filtered_df.groupby(["TweetAt", "Sentiment"]).size().unstack().fillna(0)
fig1, ax1 = plt.subplots(figsize=(10, 4))
trend_data.rolling(3).mean().plot(ax=ax1)
plt.xlabel("Date")
plt.ylabel("Tweet Count")
st.pyplot(fig1)

st.markdown("## ☁️ Word Cloud from Tweets")
all_words = " ".join(filtered_df["Cleaned_Tweet"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.imshow(wordcloud, interpolation="bilinear")
ax2.axis("off")
st.pyplot(fig2)

st.markdown("## 📊 Sentiment Distribution")
sentiment_counts = filtered_df["Sentiment"].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
ax3.axis("equal")
st.pyplot(fig3)

st.markdown("## 📄 Sample Tweets")
st.dataframe(filtered_df[["TweetAt", "Cleaned_Tweet", "Sentiment"]].sample(5))
