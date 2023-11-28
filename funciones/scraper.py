import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class Scraper:
    def __init__(self, user_input):
        self.user_input = user_input
        self.data_recom = pd.DataFrame()

    def clean_user_input(self):
        # Descargar los recursos necesarios de la libreria NLTK
        nltk.download('stopwords')
        nltk.download('punkt')
        # Convertir texto a minusculas
        text = self.user_input.lower()
        # Quitar puntuación
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Tokenizar el texto
        tokens = word_tokenize(text)
        # Quitar stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        cleaned_text = " ".join(tokens)

        return cleaned_text

    def prepare_user_input(self):
        cleaned_name = self.user_input.replace(" ", "-").lower()
        return cleaned_name

    def scraping(self):

        cleaned_text = self.clean_user_input()
        prep_clean_text = self.prepare_user_input()
        urls = ['https://listado.mercadolibre.com.ar/' + prep_clean_text]

        page_number = 50
        for i in range(0, 100, 50):
            urls.append(f"https://listado.mercadolibre.com.ar/{prep_clean_text}_Desde_{page_number + 1}_NoIndex_True")
            page_number += 50

        # Lista para almacenar lo escrapeado
        scraped_data = []

        # Iterar URL
        for i, url in enumerate(urls, start=1):
            # Traer el HTML de la pagina
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # agarro los posteos
            content = soup.find_all('li', class_='ui-search-layout__item')

            # sobre cada posteo se itera para traer el contenido
            for post in content:
                title = post.find('h2').text
                price = post.find('span', class_='andes-money-amount__fraction').text
                post_link = post.find("a")["href"]

                try:
                    brand = post.find('span', class_='ui-search-item__brand-discoverability ui-search-item__group__element').text
                except:
                    brand = '-'

                try:
                    img_link = post.find("img")["data_recom-src"]
                except:
                    img_link = post.find("img")["src"]

                try:
                    post_rvw = post.find("span", class_='ui-search-reviews__rating-number').text
                except:
                    post_rvw = '0'

                try:
                    post_rvw_amount = post.find("span", class_='ui-search-reviews__amount').text
                except:
                    post_rvw_amount = '-'

                post_data = {
                    "title": title,
                    "brand": brand,
                    "price": price,
                    "post link": post_link,
                    "image link": img_link,
                    "review": float(post_rvw),
                    "review amount": post_rvw_amount 
                }
                scraped_data.append(post_data)

        self.data_recom = pd.DataFrame(scraped_data)

def scrape_and_return_data(user_input):
    scraper = Scraper(user_input)
    scraper.scraping()
    return scraper.data_recom

