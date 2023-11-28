import time
import random
import string
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from streamlit_option_menu import option_menu
from hugchat import hugchat
from hugchat.login import Login
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds 
from scipy.sparse import csr_matrix
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.colors as colors


#1. import de dataseta
# dataset de ventas Amazon para analisis del mercado
data_amazon = pd.read_csv(r'amazon_sales.csv')

# dataset MELI
data_meli = pd.read_csv('meli.csv')

# dataset para sistema de recomendacion
data_recom = pd.read_csv('ratings.csv') 
data_recom.drop(columns='Unnamed: 0', inplace=True)


# 2. SQL Analytics
# Creo una bbdd en memoria usando sqlite
conn = sqlite3.connect(':memory:')
# Guardo el df en una tabla sql
data_amazon.to_sql('sales_data_amazon', conn, index=False)
# busco el crecimiento porcentual de la ventas mes a mes
query1 = """
WITH MonthlySales AS (
    SELECT
        Year,
        Month,
        SUM(Weekly_Sales) AS Monthly_Sales,
        LAG(SUM(Weekly_Sales)) OVER (ORDER BY Year, Month) AS Last_Month_Sales
    FROM sales_data_amazon
    GROUP BY Year, Month
)

SELECT
    Year,
    Month,
    Monthly_Sales,
    Last_Month_Sales,
    CASE
        WHEN Last_Month_Sales IS NULL THEN 0  -- To handle the first month where there is no last month sales
        ELSE (Monthly_Sales - Last_Month_Sales) / Last_Month_Sales * 100
    END AS Monthly_Growth_Percentage
FROM MonthlySales;
"""
# Crecimiento anio a anio para cada mes 
query2 = """
WITH MonthlySales AS (
    SELECT
        Year,
        Month,
        SUM(Weekly_Sales) AS Monthly_Sales
    FROM sales_data_amazon
    GROUP BY Year, Month
)

SELECT
    m1.Year ,
    m1.Month,
    m1.Monthly_Sales AS Current_Monthly_Sales,
    m2.Year AS Previous_Year,
    m2.Monthly_Sales AS Previous_Monthly_Sales,
    CASE
        WHEN m2.Monthly_Sales IS NULL THEN 0  -- To handle the first month where there is no previous year sales
        ELSE (m1.Monthly_Sales - m2.Monthly_Sales) / m2.Monthly_Sales * 100
    END AS Monthly_Growth_Percentage
FROM MonthlySales m1
LEFT JOIN MonthlySales m2 ON m1.Month = m2.Month AND m1.Year = m2.Year + 1
ORDER BY m1.Year, m1.Month;
"""
# evolucion mes a mes
monthly_sales_growth = pd.read_sql(query1, conn)
monthly_sales_growth['Year']= monthly_sales_growth['Year'].astype('str')

# evolucion mensual anio a anio
monthly_sales_comparison = pd.read_sql(query2, conn)
monthly_sales_comparison['Year']= monthly_sales_comparison['Year'].astype('str')
conn.close()


#3. Clase Scraper para web scraping
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

        # Join the tokens back into a cleaned text
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
                    brand = post.find('span', class_= 'ui-search-item__brand-discoverability ui-search-item__group__element').text
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


#5. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'EDA', 'Sistema de recomendación','Chatbot MELI', 'Chatbot LLM', 'Conclusiones'],
    )


#####################################################################################################################################


# Pagina 1 = Home
if selected == 'Home':
    st.title('E-commerce Chatbot')
    st.image("ecommerce.webp")
    st.header('Problemática y objetivos')
    st.subheader("Problema 1: 'Information Overload'")
    st.markdown("""
                El concepto de "information overload" (sobrecarga de información) se refiere a la situación en la que una persona se ve abrumada por la cantidad de información disponible, lo que dificulta su capacidad para procesar y tomar decisiones de manera efectiva.
                El exceso de información en el comercio electrónico puede abrumar a los compradores. 
    """)
    st.markdown("""Algunos problemas que trae asociado son:
                - Dificultad para tomar decisiones
                - Fatiga mental
                -Falta de claridad""")

    st.markdown("Esto afecta negativamente la experiencia del cliente y por ende ser un costo para el vendedor, y puede llevar a no encontrar el producto adecuado.")
    # Problema 2: Impacto de la inflación y crisis económica en los precios
    st.subheader("Problema 2: Inflación y crisis económica")
    st.markdown("""
    La aceleración constante de precios a nivel global, especialmente en costos de envío, ha afectado a los comerciantes en línea, generando un aumento en los precios finales de los productos.
    """)
    st.image('inflacion.png')
    
    # Problema 3: Exclusión de la población mayor en el comercio en línea
    st.subheader("Problema 3: Brecha generacional")
    st.markdown("""
                El avance tecnológico de los últimos años ha obligado a muchos a reconvertir sus hábitos, sin embargo muchas personas han quedado fuera de esta modernidad.
                La brecha generacional se manifiesta en la exclusión de adultos mayores del comercio en línea y uso de otras tecnologías modernas.
    """)
    st.image('uso_amazon_edad.png')

    # Solución: Chatbot con Web Scraping
    st.header("**Solución**: Chatbot conversacional con Web Scraping")
    st.markdown("""
    Para abordar estos problemas, surge la idea de desarrollar un producto que utilice tecnologías de **Chatbot**, **LLM** y **Web scraping**.""")
    st.subheader("**Chatbot**")
    st.markdown("""Un chatbot es un programa de computadora diseñado para interactuar con usuarios a través de conversaciones, simulando la forma en que los humanos se comunican. 
                Utiliza tecnologías como IA y NLP para comprender y responder a preguntas, realizar tareas específicas o proporcionar información en tiempo real. 
                Los chatbots se implementan en plataformas de mensajería, sitios web u otras interfaces, ofreciendo una experiencia de usuario interactiva y automatizada. """)
    st.image('Chatbot3.png')
    st.subheader("**Web Scraping**")
    st.markdown("""El web scraping es una técnica de extracción de datos que consiste en recopilar información de sitios web de manera automatizada. 
                Utiliza programas o scripts para navegar por las páginas web, analizar su estructura HTML, y extraer la información deseada, como texto, imágenes o enlaces. 
                Esta técnica permite obtener datos de manera eficiente sin necesidad de acceder manualmente a cada página.""")
    st.image("web_scrap.jpeg")
    st.subheader("**LLM**")
    st.markdown("""Los "large language models" (LLM) se refieren a modelos de inteligencia artificial diseñados para entender y generar lenguaje natural en gran escala. 
                Estos modelos son entrenados en enormes cantidades de datos textuales y utilizan arquitecturas de aprendizaje profundo, como las redes neuronales, para aprender patrones complejos en el lenguaje.""")
    st.image('LLM.png')


    st.header("Business Model Canvas")
    st.image("BMCanvas.png")


    st.header('Dataset')
    st.write('Los conjuntos de datos utilizados para este proyecto son 3:')
    st.markdown('- Amazon sales 2019-21 : evolución de las ventas de Amazon en Reino Unido')
    st.markdown('- Ratings: set de datos de ratings de productos por usuarios')
    st.markdown('- Mercado Libre: Web Scraping')
    st.markdown('Se pueden encontrar en el repositorio del proyecto: [Github Chatbot Meli](https://github.com/adilelle1/Chatbot-MELI)')
    

    

#####################################################################################################################################


# Pagina 2 = Graficos
elif selected == 'EDA':
    color_palette = colors.qualitative.Light24
    
    def ventas_por_anio(data_amazon):
        total_sales_by_year = data_amazon.groupby('Year')['Weekly_Sales'].sum().reset_index()
        total_sales_by_year = total_sales_by_year[total_sales_by_year['Weekly_Sales'] > 0]
        fig = px.bar(total_sales_by_year, x='Year', y='Weekly_Sales',
                    labels={'Weekly_Sales': 'Ventas', 'Year': 'Año'},
                    title='Ventas totales por año',
                    text='Weekly_Sales',  
                    height=500,  
                    )
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=total_sales_by_year['Year'].tolist()))
        st.plotly_chart(fig)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def crecimiento_mensual(df1):
        fig = px.bar(df1, x='Month', y='Monthly_Growth_Percentage', color='Year',
                    labels={'Monthly_Growth_Percentage': 'Porcentaje de crecimiento mensual', 'Year': 'Año'},
                    title='Crecimiento mensual mes a mes',
                    facet_col='Year',  
                    facet_col_wrap=3,  
                    height=500 
                    )
        fig.update_layout(title_font=dict(size=18))  
        st.plotly_chart(fig)

    def crecimiento_mensual_anio(df2):
        df2 = df2[df2['Year'] != 2019]
        fig = px.bar(df2, x='Month', y='Monthly_Growth_Percentage', color='Year',
                    labels={'Monthly_Growth_Percentage': 'Porcentaje de crecimiento mensual', 'Year': 'Año'},
                    title='Crecimiento mensual año a año',
                    height=500  
                    )
        fig.update_layout(title_font=dict(size=18))  
        st.plotly_chart(fig)

    def plot_aggregated_weekly_sales(data_amazon):
        # Agrupo para tener la suma total de las ventas por fecha
        weekly_sales = data_amazon.groupby('Date')['Weekly_Sales'].sum()
        # Ordeno y pongo como índice la fecha
        weekly_sales = weekly_sales.reset_index().sort_values('Date')
        fig = px.line(weekly_sales, x='Date', y='Weekly_Sales',
                    labels={'Weekly_Sales': 'Ventas', 'Date': 'Fecha'},
                    title='Evolución de las venta en el tiempo')
        st.plotly_chart(fig)

    def plot_monthly_sales_regression(data_amazon):
        # Agrupo ventas por mes y año
        monthly_sales = data_amazon.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
        # Hago numérico el mes para que mantenga el orden de mes por año
        monthly_sales['Numeric_Month'] = range(1, len(monthly_sales) + 1)
        fig = px.scatter(monthly_sales, x='Numeric_Month', y='Weekly_Sales', trendline='ols', labels={'Weekly_Sales': 'Ventas', 'Numeric_Month': 'Mes'},
                        title='Regresión lineal - ventas mensuales')
        fig.update_traces(line_shape='linear', line=dict(color='red'))
        st.plotly_chart(fig)

    golden_color = ['#FFD700']
    def show_price_distribution(data):
        fig = px.histogram(data, x='price', nbins=20, title='Distribución de precios de zapatillas',
                        labels={'price': 'Price', 'count': 'Density'},
                        color_discrete_sequence=golden_color)
        st.plotly_chart(fig)

    def show_price_review_scatter(data):
        fig = px.scatter(data, x='price', y='review', title='Relación entre precio y review score',
                        labels={'price': 'Price', 'review': 'Review Score'},
                        color_discrete_sequence=golden_color)
        st.plotly_chart(fig)

    def show_ordered_review_distribution(data):
        ordered_review_counts = data['review'].value_counts().sort_index()
        fig = px.bar(x=ordered_review_counts.index, y=ordered_review_counts.values,
                    title='Distribución de reviews',
                    labels={'x': 'Review Score', 'y': 'Count'},
                    color_discrete_sequence=golden_color)
        st.plotly_chart(fig)

    def show_top_brands(data):
        top_brands = data['brand'].value_counts().nlargest(10)
        golden_color = ['#FFD700'] * len(top_brands)  
        fig = px.bar(x=top_brands.index, y=top_brands.values,
                    title='Top 10 marcas con más publicaciones',
                    labels={'x': 'Marca', 'y': 'Número de productos publicados'},
                    color_discrete_sequence=golden_color)
        fig.update_layout(xaxis=dict(tickangle=45, tickmode='array', tickvals=top_brands.index, ticktext=top_brands.index))
        st.plotly_chart(fig)


    if __name__ == '__main__':
        st.title('Análisis exploratorio de datos')
        st.header('Ventas Amazon UK 2019-2021')
        st.subheader('Ventas por año')
        ventas_por_anio(data_amazon)

        st.subheader('Evolución mensual de las ventas')
        crecimiento_mensual(monthly_sales_growth)
        #crecimiento_mensual_anio(monthly_sales_comparison)

        st.subheader('Evolución de las ventas en el tiempo')
        plot_aggregated_weekly_sales(data_amazon)
        plot_monthly_sales_regression(data_amazon)

        st.header('Mercado Libre web scraping - zapatillas')
        show_price_distribution(data_meli)
        show_ordered_review_distribution(data_meli)
        show_price_review_scatter(data_meli)
        show_top_brands(data_meli)

        st.markdown('''
        <style>
        [data_recom-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)


#####################################################################################################################################

# Pagina 3 = Sistema de recom
elif selected == 'Sistema de recomendación':
    def sistemas_de_recomendacion():
        st.title('Sistema de recomendación')
        st.write('Las empresas de comercio electrónico como Mercado Libre utilizan diversos sistemas de recomendación para proporcionar sugerencias a los clientes. \
                 Amazon actualmente emplea un filtro colaborativo de tipo item-item, que se adapta a conjuntos de datos masivos y genera un sistema de recomendación de alta calidad en tiempo real. \
                 Para este proyecto, se veran algunas formas de recomendar más sencillas para finalmente llegar a un sistema de filtrado de información que busca predecir la calificación de productos para ver las preferencias en las que el usuario está interesado.')

        st.header("Introducción a los sistemas de recomendación")
        st.write("En este mundo moderno, donde estamos sobrecargados de datos, no es tan sencillo para el usuario extraer valor de toda la informaación que recibe. \
                 Para ayudar al usuario a encontrar información sobre el producto que está buscando, se desarrollan los sistemas de recomendación.")
        st.header("¿Qué problemas puede resolver un sistema de recomendación?")
        st.write("Puede ayudar al usuario a encontrar el producto adecuado.")
        st.write("Puede aumentar la participación del usuario. Por ejemplo, hay un 40% más de clics en las noticias de Google debido a las recomendaciones.")
        st.write("Ayuda a los proveedores de productos a entregar los artículos al usuario adecuado. En Amazon, el 35% de los productos se venden gracias a las recomendaciones.")
        st.write("Ayuda a personalizar el contenido. En Netflix, la mayoría de las películas vistas son a través de recomendaciones.")

        st.header("Tipos de recomendaciones")
        st.image("recommenders.png")
        #st.write("Principalmente hay 6 tipos de sistemas de recomendación:")
        #st.markdown("- **Sistemas basados en popularidad:** Funcionan recomendando elementos vistos y comprados por la mayoría de las personas y que tienen altas calificaciones. No son recomendaciones personalizadas.")
        #st.markdown("- **Modelo basado en clasificación:** Funciona comprendiendo las características del usuario y aplicando el algoritmo de clasificación para decidir si el usuario está interesado o no en el producto.")
        #st.markdown("- **Recomendaciones basadas en contenido:** Se basa en la información sobre el contenido del artículo en lugar de en las opiniones del usuario. La idea principal es que si al usuario le gusta un artículo, entonces le gustará el 'otro' artículo similar.")
        #st.markdown("- **Filtrado colaborativo:** Se basa en la suposición de que a las personas les gustan productos similares a otros que ya les gustaron, y productos que son valorados por otras personas con gustos similares. \
        #            Principalmente hay dos tipos: \n\t\ta) Usuario-Usuario \n\t\tb) Artículo-Artículo")
        #st.markdown("- **Enfoques híbridos:** Este enfoque combina filtrado colaborativo, filtrado basado en contenido y otros enfoques.")
        #st.markdown("- **Minería de reglas de asociación:** Las reglas de asociación capturan las relaciones entre elementos basadas en sus patrones de co-ocurrencia en las transacciones.")

        st.header("Información del dataset:")
        st.write("A modo de ejemplo, se utiliza un conjunto de datos de pocas dimensiones donde se tiene información de usuarios y la forma en que 'ratearon' diferentes productos.")
        st.write("Por otro lado, se hace un recorte del dataset para tomar solo aquellos usuarios con más de 50 valoraciones de productos. Esto le dará más robustez al análisis y nos permite trabajar con un dataset de 125 mil registros (antes 7 millones).")
        st.markdown("- **userId**: Cada usuario identificado con un ID único.")
        st.markdown("- **productId**: Cada producto identificado con un ID único.")
        st.markdown("- **Rating:** Calificación del producto por el usuario.")
        st.dataframe(data_recom.head())
        fig = px.histogram(data_recom, x='rating', nbins=5, title='Distribución de ratings',
                        labels={'rating': 'Rating', 'count': 'Count'},
                        color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig)






        ####### PRIMERA RECOM ##########
        st.header("Recomendación por productos calientes (muy vendidos/recomendados)")
        st.write("Se realiza un análisis de calificaciones de productos calculando el rating promedio y la cantidad de calificaciones por producto. \
                 Con esta información, se encuentran los productos con un número mínimo de interacciones (ratings) y se los ordena por rating promedio. \
                 A partir de este análisis es posible recomendar los productos mejor valorados con suficientes interacciones.")
        # Rating pormedio por producto
        average_rating = data_recom.groupby('prod_id').mean()['rating']
        # Cantidad de ratings por producto
        count_rating = data_recom.groupby('prod_id').count()['rating']
        final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})
        final_rating = final_rating.sort_values(by='avg_rating',ascending=False)
        # traer los top_n productos con cierto nivel minimo de interacciones 
        def top_n_products(final_rating, top_n, interacciones_minimas):
            recommendaciones = final_rating[final_rating['rating_count']>interacciones_minimas]
            # ordenar los rdos
            recommendaciones = recommendaciones.sort_values('avg_rating',ascending=False)
            return recommendaciones.index[:top_n]

        min_interactions_str  = st.text_input('Ingresa el número mínimo de interacciones', value='', key='min_interactions')

        if min_interactions_str  != '': 
            try:
                min_interactions = int(min_interactions_str)
                hot_prods = list(top_n_products(final_rating, 5, min_interactions))
                st.write('**Productos más recomendados:**')
                for prod in hot_prods:
                    st.write(f'**{prod}**')
                    
                    avg_rating_value = final_rating.loc[final_rating.index == prod, 'avg_rating'].values[0]
                    rating_count_value = final_rating.loc[final_rating.index == prod, 'rating_count'].values[0]
                    
                    st.markdown(f'- Rating promedio: {round(avg_rating_value)}')
                    st.markdown(f'- Cantidad de ratings: {rating_count_value}')

            except ValueError:
                st.warning('Por favor ingresa un numero entero.')
            except IndexError:
                st.warning('Por favor ingresa un numero entero.')





        ####### SEGUNDA RECOM ##########
        st.header("Recomendación por Collaborative Filtering")
        st.markdown("Se basa en la suposición de que a las personas les gustan productos similares a otros que les gustaron en el pasado, y produtos que son apreciadas por otras personas con gustos similares.")
        st.markdown(" Principalmente, existen dos tipos:")
        st.markdown("- Usuario-Usuario.")
        st.markdown("- Item-Item.")

        st.subheader("Basado en usuarios")
        st.markdown("""
                Se crea una matriz cruzando productos y usuarios, teniendo como valores los ratings que cada usuario le dio a cada producto. 
                Se calcula para determinar la densidad de valores no vacíos en la matriz.
                La densidad proporciona una medida de qué tan completa está la matriz en términos de valores no vacíos en comparación con su capacidad total.
                    """)
        
        st.markdown("""A través de una función que recibe el id del usuario se obtienen los usuarios más similares. 
                    Se utiliza cosine_similarity para calcular el grado de similitud.""")
        
        st.image('cosine-similarity-vectors.jpg')
        st.image('similarity-formula.png')

        st.markdown("""Luego se utiliza el usuario más similar para encontrar cuáles fueron sus productos favoritos (mejor rating).""")

        matriz_ratings_final = data_recom.pivot(index = 'user_id', columns ='prod_id', values = 'rating').fillna(0)
        matriz_ratings_final['user_index'] = np.arange(0, matriz_ratings_final.shape[0])
        matriz_ratings_final.set_index(['user_index'], inplace=True)
        def similar_users(user_index, interactions_matrix):
            similarity = []

            for user in range(interactions_matrix.shape[0]):
                # le pasamos un usuarioid y buscamos la similitud con el resto
                sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])[0][0]

                # guardo los rdos en formato tupla dentro de una lista
                similarity.append((user, sim))

            similarity.sort(key=lambda x: x[1], reverse=True)
            # Extract user index and similarity score from the sorted list
            most_similar_users = [tup[0] for tup in similarity]#extraer el usuario de la tupla
            similarity_score = [tup[1] for tup in similarity]# extraer el score de similitud
            # quito de la lista el mismo usuario que busque
            most_similar_users.remove(user_index)
            similarity_score.remove(similarity_score[0])
            # Guardo en df
            result_df = pd.DataFrame({'UserIndex': most_similar_users, 'SimilarityScore': similarity_score})
            return result_df

        user_id_input  = st.text_input('Ingresa el id del usuario', value='', key='id_usuario')

        if user_id_input  != '': 
            try:
                user_id_input_int = int(user_id_input)
                usuarios_similares = similar_users(user_id_input_int,matriz_ratings_final)

                st.write('**Usuarios similares:**')
                st.dataframe(usuarios_similares.head(5))

            except ValueError:
                st.warning('Por favor ingresa un numero entero.')
            except IndexError:
                st.warning('Por favor ingresa un numero entero.')

        #Usando los resultados de similitud de usuario creamos una funcion de recomendacion de productos
        def recommendations(user_index, num_of_products, interactions_matrix):
            # Uso la funcion de usuarios similares, tomo el primero
            most_similar_users = similar_users(user_index, interactions_matrix).loc[0]
            # Busco los productos del usuario similar
            product_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
            recommendations = []
            observed_interactions = product_ids.copy()
            for similar_user in most_similar_users:
                if len(recommendations) < num_of_products:
                    # Busco productos rateados por el usuario similar pero no por el user_id
                    similar_user_prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
                    recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
                    observed_interactions = observed_interactions.union(similar_user_prod_ids)
                else:
                    break
            return recommendations[:num_of_products]
        
        
        if user_id_input  != '': 
            try:
                prod_recomen_user_sim = recommendations(user_id_input_int, 5, matriz_ratings_final)
                st.write('**Productos recomendados:**')
                for prod in prod_recomen_user_sim:
                    st.write(f'- **{prod}**')
                
            except ValueError:
                st.warning('Por favor ingresa un numero entero.')
            except IndexError:
                st.warning('Por favor ingresa un numero entero.')






        ####### TERCERA RECOM ##########
        st.subheader("Modelo basado en Collaborative Filtering: Singular Value Decomposition")
        st.write("Como la matriz de interacción para este conjunto de datos es altamente dispersa (más del 99% de los valores son 0).\
                  Se utiliza la técnica SVD, en conjunto con la predicción de ratings faltantes de los productos.")
        st.markdown("**SVD** (Descomposición de Valores Singulares, por sus siglas en inglés) es una técnica matemática en álgebra lineal que descompone una matriz en tres matrices más simples. ")
        st.markdown("Siendo M la matriz dada, usando SVD se descompone de la siguiente manera")
        st.latex(r'M = U \Sigma V^T')
        st.markdown("""
                Donde:
                - (M) es la matriz original,
                - (U) es una matriz unitaria (de usuarios),
                - (Sigma) es una matriz diagonal que contiene los valores singulares de M,
                - (V^T) es la traspuesta de la matriz unitaria (de productos).
                """)
        
        st.image('SVD_Expl.jpg')
        st.image('svd_UxV.png')


        final_ratings_sparse = csr_matrix(matriz_ratings_final.values)
        # Singular Value Decomposition
        U, s, Vt = svds(final_ratings_sparse, k = 50) # k es el numero de features latentes
        # construir un array diagonal en SVD
        sigma = np.diag(s)
        users_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        # Ratings predichos
        preds_df = pd.DataFrame(abs(users_predicted_ratings), columns = matriz_ratings_final.columns)
        preds_df.head()
        preds_matrix = csr_matrix(preds_df.values)
        def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
            # Tomo ratings reales y predichos de las matrices
            user_ratings = interactions_matrix[user_index,:].toarray().reshape(-1)
            user_predictions = preds_matrix[user_index,:].toarray().reshape(-1)

            # Creo un df con ratings reales y predichos en columnas
            temp = pd.DataFrame({'user_ratings': user_ratings, 'user_rating_predictions': user_predictions})
            temp['Recommended Products'] = np.arange(len(user_ratings))
            temp = temp.set_index('Recommended Products')

            # Filtro ese df donde los ratings reales son 0, es decir donde el usuario no interactuo con ese producto
            temp = temp.loc[temp.user_ratings == 0]

            # Recomendando en base a la prediccion de los productos mejor rankeados para un usuario
            temp = temp.sort_values('user_rating_predictions',ascending=False) # ordenar por prediccion descendiente 
            return temp
        
        user_id_2  = st.text_input('Ingresa el id del usuario', value='', key='id_usuario_prediccion')

        if user_id_2  != '': 
            try:
                user_id_2_int = int(user_id_2)
                prod_recomen_pred_rating = recommend_items(user_id_2_int, final_ratings_sparse, preds_matrix,5)
                st.write('**Productos recomendados en base a predicción de rating:**')
                #for prod in prod_recomen_user_sim:
                #    st.write(f'- **{prod}**')
                st.dataframe(prod_recomen_pred_rating['user_rating_predictions'].head(10))
                st.subheader("Evaluación de la predicción")
                st.markdown("Se toman los valores reales (ratings) y se calcula un promedio de cada artiuclo.")
                st.markdown("Se toman los valores predichos (ratings) y se calcula un promedio de cada artiuclo.")
                st.markdown("Se toman los valores reales (ratings) y se calcula un promedio de cada artiuclo.")
                matriz_ratings_final['user_index'] = np.arange(0, matriz_ratings_final.shape[0])
                matriz_ratings_final.set_index(['user_index'], inplace=True)
                average_rating = matriz_ratings_final.mean()
                avg_preds=preds_df.mean()
                rmse_df = pd.concat([average_rating, avg_preds], axis=1)
                rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
                st.dataframe(rmse_df.head())
                RMSE= mean_squared_error(rmse_df['Avg_actual_ratings'], rmse_df['Avg_predicted_ratings'], squared=False)
                st.markdown(f'**RMSE SVD Model** = {RMSE}')
                st.markdown('Un valor más bajo de RMSE indica que el modelo tiene un mejor rendimiento en términos de predicciones precisas.')
                
            except ValueError:
                st.warning('Por favor ingresa un numero entero.')
            except IndexError:
                st.warning('Por favor ingresa un numero entero.')

        




    if __name__ == '__main__':
        sistemas_de_recomendacion()


#####################################################################################################################################


# Pagina 4 = chatbot 1
elif selected == 'Chatbot MELI':
    def chatbot_meli():
        user_input = []

        st.title("Chatbot MELI")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.user_inputs = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input('Comenzá a chatear con Marquitos!'):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.user_inputs.append(prompt)
                with st.chat_message("user"):
                    st.markdown(prompt)
                user_input.append(prompt)

        # Initialize questions for the sales conversation
        sales_questions = [
            "Hola! Soy Marquitos-G 😎, qué producto estás buscando?",
            "Qué tipo de zapatillas estás buscando? (running, futbol, tenis, basquet, etc.)",
            "Cuál es tu talle?",
            "Qué color estás buscando?",
            "Qué marca te gustaría?",
        ]


        # Check if the conversation is at the beginning
        if not st.session_state.messages:
            # Display the initial random response
            assistant_response = random.choice(
                [
                    "Hola! Soy Marquitos-G 😎, qué producto estás buscando?"
                ]
            )
            

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Simulate typing the initial response
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Continue the sales conversation by asking questions
            if len(st.session_state.messages) < len(sales_questions) * 2:
                # Calculate the index of the current question
                question_index = len(st.session_state.messages) // 2
                # Display the next question
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    # Simulate typing the question
                    for chunk in sales_questions[question_index].split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                # Add assistant question to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                
            else:
                # If all questions have been asked, end the conversation
                with st.chat_message('assistant'):
                
                    with st.spinner(f"Buscando...🤔"):
                        user_input_string = " ".join(st.session_state.user_inputs)
                        scraper = Scraper(user_input_string)
                        scraper.scraping()
                        prod1 = scraper.data_recom.iloc[0]
                        prod2 = scraper.data_recom.iloc[1]
                        prod3 = scraper.data_recom.iloc[2]
                        
                        st.session_state.messages.append({"role": "assistant", "content": f"{prod1['title']} -- ${prod1['price']} -- {prod1['post link']}"})
                        st.session_state.messages.append({"role": "assistant", "content": f"{prod2['title']} -- ${prod2['price']} -- {prod2['post link']}"})
                        st.session_state.messages.append({"role": "assistant", "content": f"{prod3['title']} -- ${prod3['price']} -- {prod3['post link']}"})
                        st.markdown(user_input_string)
                        st.markdown(f"**Producto 1**\
                                    \n Título:{prod1['title']}\
                                    \n Marca:{prod1['brand']}\
                                    \n Precio: {prod1['price']}\
                                    \n Enlace al sitio: [link]({prod1['post link']})\
                                    \n Puntuación: {prod1['review']}\
                                    \n Cantidad de reviews: {prod1['review amount']}")
                        st.markdown(f"**Producto 2**\
                                    \n Título:{prod2['title']}\
                                    \n Marca:{prod2['brand']}\
                                    \n Precio: {prod2['price']}\
                                    \n Enlace al sitio: [link]({prod2['post link']})\
                                    \n Puntuación: {prod2['review']}\
                                    \n Cantidad de reviews: {prod2['review amount']}")
                        st.markdown(f"**Producto 3**\
                                    \n Título:{prod3['title']}\
                                    \n Marca:{prod3['brand']}\
                                    \n Precio: {prod3['price']}\
                                    \n Enlace al sitio: [link]({prod3['post link']})\
                                    \n Puntuación: {prod3['review']}\
                                    \n Cantidad de reviews: {prod3['review amount']}")
                        
            
              

                with st.chat_message("assistant"):
                    st.markdown("Espero que estos productos te sean de utilidad! Querés buscar algo más?")
                    
                    # Add buttons for user response
                    user_response = st.radio("Selecciona una opción:", ["No", "Sí"])
                    
                    if user_response == "No":
                        # If user selects "No", end the conversation with a goodbye message
                        st.session_state.messages.append({"role": "user", "content": user_response})
                        st.session_state.messages.append({"role": "assistant", "content": "¡Hasta luego! 👋"})
                        st.markdown("¡Hasta luego! 👋")
                        
                    elif user_response == "Sí":
                        # If user selects "Sí", reset the conversation
                        st.session_state.messages = []
                        st.markdown("¡Perfecto! Comencemos de nuevo.")
                        
                        
                        


    if __name__ == '__main__':
        chatbot_meli()


#####################################################################################################################################


# Pagina 5 = Modelo
elif selected == 'Chatbot LLM':
    def chatbot_llm():
        st.title("🤗💬 HugChat")

        with st.sidebar:
            st.title(f"{''':handshake:'''} Chatbot Meli {''':handshake:'''}")
            if 'hf_email' not in st.session_state or 'hf_pass' not in st.session_state:
                st.write("⚠️ Debes registrarte en HugginFace para usar esta app. Puedes registrarte aquí [🤗](https://huggingface.co/join).")
                hf_email = st.text_input('Ingrese su E-mail:', type='default')
                hf_pass = st.text_input('Ingrese su password:', type='password')
                if not (hf_email and hf_pass):
                    st.warning('Por favor ingresa tus credenciales!', icon='⚠️')
                else:
                    st.success('Listo! Comienza a conversar con Meli, tu asistente', icon='👉')
            st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
            # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])


        # Function for generating LLM response
        def generate_response(prompt_input, email, passwd):
            # Hugging Face Login
            sign = Login(email, passwd)
            cookies = sign.login()
            # Create ChatBot                        
            chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
            return chatbot.chat(prompt_input)

        # User-provided prompt
        if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt, hf_email, hf_pass) 
                    st.write(response) 
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

    if __name__ == '__main__':
        chatbot_llm()


#####################################################################################################################################


# Pagina 5 = Conclusiones
elif selected == 'Conclusiones':
    st.title("Conclusiones")
    st.header("KPIs de medición")
    st.markdown(""" 
                - **Tasa de Conversión** (Conversion Rate): Proporción de usuarios que realizaron una compra en comparación con el total de usuarios que interactuaron con el chatbot.
                - **Ingresos Generados**: Ingresos directamente atribuibles a las recomendaciones del chatbot. 
                - **Interacción Promedio** del Usuario: Cantidad de interacciones que un usuario tiene con el chatbot durante una sesión.
                - **Tiempo de Respuesta**: Rapidez con la que el chatbot responde a las consultas de los usuarios. 
                - **Retención de Usuarios**: Frecuencia con la que los usuarios regresan al chatbot para obtener más recomendaciones. 
                - **Análisis de sentimiento**: Análisis usndo el feedback de los usuarios después de interactuar con el chatbot. 
                - **Número de Transacciones por Usuario**: Cantidad de transacciones realizadas por cada usuario. 
                - **Porcentaje de Abandono**: Cantidad de usuarios que abandonan la interacción con el chatbot antes de completar una transacción.
                - **Costo por Adquisición (CPA)**: Si hay inversión en la promoción del chatbot, medir cuánto cuesta adquirir un nuevo cliente a través del chatbot.
            """)      

    st.header("Perspectivas futuras")
    st.markdown("""Para avanzar en el armado del producto final será necesario sortear los desafíos que apareceran cómo:""")
    st.markdown("""
                - Acuerdos de uso de datos de sitios web
                - Escalabilidad del proyecto
                - Costos de desarrollo
                - Aspectos legales del uso de datos 
                """)
    st.markdown(""" 
                Usando una tecnología de chatbot conversacional, la aplicación de técnicas de web scraping y sistemas de recomendación podrían representar un futuro prometedor en la mejora de la experiencia de compra en línea. 
                Por otro lado, el potencial de crecimiento utilizando Modelos de Lenguaje Extenso brindan un gran horizonte de crecimiento para la solución \
                 pensada en este proyecto, donde el usuario no solo recibiría respuestas automáticas creadas manualmente por los desarrolladores, sino que podría tener una conversación más real donde se le den recomendaciones de los mejores productos según diferentes fuentes de internet, ayuda personalizada según las últimas tendencias, o hasta incluso un juego de roles donde el modelo se pone en el lugar de un vendedor de zapatillas para mejorar la atención al cliente.
                """)
    
