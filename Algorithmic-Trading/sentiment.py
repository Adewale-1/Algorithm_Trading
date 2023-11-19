import openai
import feedparser
from urllib.parse import quote
import os
from dotenv import load_dotenv


load_dotenv()
# Use the correct OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to analyze sentiment using chat model GPT-3.5-turbo


def analyze_sentiment_with_gpt(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        sentiment = response.choices[0].message['content'].strip()
        return sentiment
    except openai.error.OpenAIError as e:
        print(f"OpenAI Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def fetch_google_news_rss(search_query):
    feed_url = f"https://news.google.com/rss/search?q={quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(feed_url)


def get_sentiment_for_google_news_articles(search_query):
    news_feed = fetch_google_news_rss(search_query)

    # Iterate through entries (news articles)
    for entry in news_feed.entries:
        # Format messages for chat model
        messages = [
            {"role": "system", "content": "You are an AI that can analyze news article sentiments."},
            {"role": "user", "content": f"Title: {entry.title}"},
            {"role": "user", "content": f"Summary: {entry.summary}"}
        ]

        # Get sentiment analysis from the model
        sentiment_analysis = analyze_sentiment_with_gpt(messages)

        # Print the title, summary, and sentiment analysis
        print(f"Title: {entry.title}")
        print(f"Summary: {entry.summary}")
        print(f"Sentiment Analysis: {sentiment_analysis}")
        print('-' * 80)


if __name__ == "__main__":
    search_query = "EUR/USD"
    get_sentiment_for_google_news_articles(search_query)
