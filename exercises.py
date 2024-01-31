import os
import requests
import validators
from bs4 import BeautifulSoup

# Define the base URL to be scraped
BASE_URL = "https://www.harvard.edu"

# Function to scrape content from a given URL
def scrape_content(url):
    # Check if the URL is valid
    if not validators.url(url):
        print("Invalid URL:", url)
        return None

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the HTTP connection is successful (status code 200)
    if response.status_code == 200:
        print("HTTP connection is successful! for the URL:", url)
        return response
    else:
        print("HTTP connection is NOT successful! for the URL:", url)
        return None

# Function to save HTML content to a file
def save_html(to_where, text, name):
    file_name = f"{name}.html"
    # Write the HTML content to a file in the specified directory
    with open(os.path.join(to_where, file_name), "w", encoding="utf-8") as f:
        f.write(text)

# Function to create a mini dataset by scraping multiple URLs
def create_mini_dataset(to_where, url_list):
    # Iterate through the list of URLs
    for i, url in enumerate(url_list):
        # Scrape the content from the URL
        content = scrape_content(url)

        # If content is obtained, save it as an HTML file
        if content is not None:
            save_html(to_where, content.text, str(i))
        else:
            pass
    print("Mini dataset is created!")

# Define the folder name for the mini dataset
folder = "mini_dataset"
# Get the absolute path to the folder
path = os.path.join(os.getcwd(), folder)
# Check if the folder exists, if not, create it
if not os.path.exists(folder):
    os.mkdir(folder)

# List of URLs to be scraped
URL_LIST = [
    "https://www.harvard.edu",
]

# Call the function to create the mini dataset
create_mini_dataset(path, URL_LIST)
