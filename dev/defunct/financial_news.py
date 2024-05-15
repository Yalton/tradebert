import requests
from bs4 import BeautifulSoup

def extract_financial_news():
    # Define the URL to extract news from
    url = 'https://www.marketwatch.com/newsviewer'

    # Send an HTTP request to the URL
    response = requests.get(url)

    # Extract the HTML content
    content = response.content

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Extract the news articles from the parsed content
    articles = soup.find_all('div', {'class': 'article__content'})

    # Filter out non-financial news articles
    filtered_articles = []

    for article in articles:
        if 'marketwatch.com/story/' in article.find('a')['href']:
            filtered_articles.append(article)

    return filtered_articles