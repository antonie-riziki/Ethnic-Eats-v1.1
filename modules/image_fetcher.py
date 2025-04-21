import requests
from bs4 import BeautifulSoup

def fetch_food_image(food_name):
    search_url = f"https://www.google.com/search?hl=en&q={food_name.replace(' ', '+')}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    image_tags = soup.find_all("img")

    if len(image_tags) > 1:
        return image_tags[1]["src"]
    else:
        return 'https://via.placeholder.com/200x300.png?text=No+Image+Available'
