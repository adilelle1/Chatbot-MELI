import streamlit as st
import plotly.express as px


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
