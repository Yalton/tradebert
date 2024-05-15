# Import necessary libraries
import requests
from bs4 import BeautifulSoup

# Define search terms
search_terms = "current cryptocurrency market trends"

# Fetch search results
url = f"https://www.google.com/search?q={search_terms}&tbm=nws"
res = requests.get(url)
res.raise_for_status()

# Parse search results
soup = BeautifulSoup(res.text, "html.parser")
search_results = soup.select(".dbsr")
top_ten_results = [result.a.text for result in search_results[:10]]

# Print top 10 results
for i in range(len(top_ten_results)):
    print(f"{i+1}. {top_ten_results[i]}")
