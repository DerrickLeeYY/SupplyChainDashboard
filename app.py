import streamlit as st
import pandas as pd

st.set_page_config(page_title="Column Load Test", layout="wide")

FILE = "cleaned_m5_dashboard_data.parquet"

st.title("Column Load Test")

column_sets = {
    "1_date_revenue": ["date", "revenue"],
    "2_basic_sales": ["date", "sales", "revenue", "weekday"],
    "3_filters": ["date", "sales", "revenue", "weekday", "state_id", "store_id", "cat_id"],
    "4_products": ["date", "sales", "revenue", "weekday", "state_id", "store_id", "cat_id", "item_id"],
    "5_price": ["date", "sales", "revenue", "weekday", "state_id", "store_id", "cat_id", "item_id", "sell_price"],
}

choice = st.selectbox("Choose column set to test", list(column_sets.keys()))

cols = column_sets[choice]

st.write("Trying to load columns:", cols)

df = pd.read_parquet(FILE, columns=cols)

st.success("Loaded successfully")
st.write("Shape:", df.shape)
st.write("Memory MB:", df.memory_usage(deep=True).sum() / 1024 / 1024)
st.dataframe(df.head())