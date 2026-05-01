import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Parquet Test", layout="wide")

st.title("Parquet Test")

df = pd.read_parquet(
    "cleaned_m5_dashboard_data.parquet",
    columns=["date", "revenue"]
)

st.write("Loaded successfully")
st.write(df.shape)

df["date"] = pd.to_datetime(df["date"])

monthly_revenue = (
    df.groupby(df["date"].dt.to_period("M"))["revenue"]
    .sum()
    .reset_index()
)

monthly_revenue["date"] = monthly_revenue["date"].dt.to_timestamp()

fig = px.line(monthly_revenue, x="date", y="revenue", title="Revenue by Month")
st.plotly_chart(fig, use_container_width=True)