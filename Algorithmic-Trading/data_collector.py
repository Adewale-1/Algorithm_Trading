import yfinance as yf
import pandas as pd
from datetime import datetime
import sys
import time
import logging
import openai
import feedparser
from urllib.parse import quote
import os
from dotenv import load_dotenv
import re
from html import unescape
import requests
from bs4 import BeautifulSoup


load_dotenv()
# Use the correct OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(filename='data_collector.log', filemode='a',
                    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def clean_summary(summary):
    # Parse the provided HTML summary to extract the URL
    soup = BeautifulSoup(summary, 'html.parser')
    link = soup.find('a', href=True)

    # Now 'link' contains the first (or only) 'a' tag with an 'href' attribute
    if link:
        article_url = link['href']
        print(article_url)  # Output the URL
    else:
        article_url = None
        print("No URL found.")  # Handle the case where no link was found

    return article_url


def analyze_sentiment_with_gpt(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        sentiment = response.choices[0].message['content'].strip()
        return sentiment
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI Error: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


def fetch_google_news_rss(search_query):
    feed_url = f"https://news.google.com/rss/search?q={quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(feed_url)


def get_article_text(url):
    # Send HTTP GET request to the URL
    response = requests.get(url)

    # If the response status code is 200 OK, parse the content
    if response.status_code == 200:
        article_html = response.text
        soup = BeautifulSoup(article_html, 'html.parser')

        # This will vary significantly depending on the website structure
        # I might need to inspect the HTML and find the actual tags/class names/etc.
        # that contain the article text
        article_body = soup.find('div', {'class': 'article-body'})
        if article_body:
            return article_body.get_text(strip=True)
    return None


def get_sentiment_for_google_news_articles(currency_pair):
    search_query = f"{currency_pair} forex"
    news_feed = fetch_google_news_rss(search_query)

    sentiments = []
    for entry in news_feed.entries[:5]:
        # Clean the HTML tags from the summary text
        # article_text = clean_html(entry.summary)
        messages = [
            {"role": "system", "content": "Can you analyse this article and decide if it is a Positive , Negative or Neutral"},
            {"role": "user", "content": f"Title: {entry.title}"},
            {"role": "user",
                "content": f"Summary: {get_article_text(clean_summary(entry.summary))}"}
        ]
        print(get_article_text(clean_summary(entry.summary)))

        sentiment_analysis = analyze_sentiment_with_gpt(messages)
        sentiments.append(sentiment_analysis)

    # Default to "Neutral" if no sentiments found
    # return sentiments if sentiments else ["Neutral"] * len(news_feed.entries)

    return sentiments


def download_data(currency_pair, start_date, end_date, time_interval):
    data = yf.download(currency_pair, start=start_date,
                       end=end_date, interval=time_interval, progress=False)

    if data.empty:
        raise ValueError(
            f"No data found for {currency_pair}, {start_date} to {end_date}, {time_interval}")

    if data.isnull().values.any():
        raise ValueError("Data contains NaN values.")

    return data


def download_data_and_sentiment(currency_pair, start_date, end_date, time_interval):
    data = download_data(currency_pair, start_date, end_date, time_interval)
    sentiments = get_sentiment_for_google_news_articles(currency_pair)

    # Assuming one sentiment analysis value per period
    # Default to "Neutral" if no sentiments found
    data['Sentiment'] = sentiments[0] if sentiments else "Neutral"

    filename = f"{currency_pair}_{start_date}_to_{end_date}_{time_interval}_sentiment.csv"
    data.to_csv(filename)

    logging.info(f"Data with sentiment saved to {filename}")
    print(f"Data with sentiment saved to {filename}")


if __name__ == "__main__":

    currency_pair = "EURUSD=X"
    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    time_interval = "1h"  # Daily interval
    download_data_and_sentiment(
        currency_pair, start_date, end_date, time_interval)
