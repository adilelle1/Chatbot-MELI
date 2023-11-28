import sqlite3
import pandas as pd

def create_and_query_sql_analytics(data_frame):
    # Create a SQLite in-memory database
    conn = sqlite3.connect(':memory:')
    
    # Guardo los datos en una tabla sql
    data_frame.to_sql('sales_data_amazon', conn, index=False)
    
    # Query 1: Monthly Sales Growth Percentage
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
            WHEN Last_Month_Sales IS NULL THEN 0
            ELSE (Monthly_Sales - Last_Month_Sales) / Last_Month_Sales * 100
        END AS Monthly_Growth_Percentage
    FROM MonthlySales;
    """
    
    # Query 2: Yearly Monthly Sales Growth Percentage
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
            WHEN m2.Monthly_Sales IS NULL THEN 0
            ELSE (m1.Monthly_Sales - m2.Monthly_Sales) / m2.Monthly_Sales * 100
        END AS Monthly_Growth_Percentage
    FROM MonthlySales m1
    LEFT JOIN MonthlySales m2 ON m1.Month = m2.Month AND m1.Year = m2.Year + 1
    ORDER BY m1.Year, m1.Month;
    """
    
    # Ejecuto queries y los guardo en df
    monthly_sales_growth = pd.read_sql(query1, conn)
    monthly_sales_growth['Year'] = monthly_sales_growth['Year'].astype('str')

    monthly_sales_comparison = pd.read_sql(query2, conn)
    monthly_sales_comparison['Year'] = monthly_sales_comparison['Year'].astype('str')
    
    conn.close()
    
    return monthly_sales_growth, monthly_sales_comparison

