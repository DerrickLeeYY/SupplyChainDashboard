import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

import os
import sys

st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet(
        "cleaned_m5_dashboard_data.parquet",
        columns=[
            "date", "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "sales", "sell_price", "revenue", "weekday", "is_event"
        ]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("int16")
    df["sell_price"] = df["sell_price"].astype("float32")
    df["revenue"] = (df["sales"] * df["sell_price"]).astype("float32")
    df["is_event"] = df["is_event"].astype("bool")

    df["year"] = df["date"].dt.year
    df["month_start"] = df["date"].values.astype("datetime64[M]")

    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]:
        df[col] = df[col].astype("category")

    return df


def bar_chart(data, x, y, title):
    fig = px.bar(data, x=x, y=y, orientation="h", title=title)
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title()
    )
    st.plotly_chart(fig, use_container_width=True)


def top_bar(df, group_col, value_col, title, n=10):
    temp = (
        df.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(n)
        .sort_values(value_col, ascending=True)
    )
    bar_chart(temp, value_col, group_col, title)


df = load_data()

st.title("MA6721 Group 1 CA3 M5 Sales Dashboard")
st.write("This dashboard uses a snapshot of the data from 1st Feburary 2011 to 24th April 2016. It treats today as 24th April 2016.")

# Sidebar filters
st.sidebar.header("Filters")

month_options = sorted(df["month_start"].unique())
month_labels = [pd.to_datetime(m).strftime("%b %Y") for m in month_options]

start_month_label = st.sidebar.selectbox("Start Month", month_labels, index=0)
start_idx = month_labels.index(start_month_label)

end_month_label = st.sidebar.selectbox(
    "End Month",
    month_labels[start_idx:],
    index=len(month_labels[start_idx:]) - 1
)

start_month = pd.to_datetime(month_options[start_idx])
end_month = pd.to_datetime(month_options[month_labels.index(end_month_label)]) + pd.offsets.MonthEnd(0)

state_filter = st.sidebar.multiselect(
    "Select State",
    sorted(df["state_id"].unique()),
    default=sorted(df["state_id"].unique())
)

store_filter = st.sidebar.multiselect(
    "Select Store",
    sorted(df["store_id"].unique()),
    default=sorted(df["store_id"].unique())
)

cat_filter = st.sidebar.multiselect(
    "Select Category",
    sorted(df["cat_id"].unique()),
    default=sorted(df["cat_id"].unique())
)

filtered_df = df[
    (df["date"] >= start_month) &
    (df["date"] <= end_month) &
    (df["state_id"].isin(state_filter)) &
    (df["store_id"].isin(store_filter)) &
    (df["cat_id"].isin(cat_filter))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# KPIs
cols = st.columns(4)
cols[0].metric("Total Sales", f"{filtered_df['sales'].sum():,.0f}")
cols[1].metric("Total Revenue", f"${filtered_df['revenue'].sum():,.2f}")
cols[2].metric("Avg Daily Sales", f"{filtered_df.groupby('date')['sales'].sum().mean():,.2f}")
cols[3].metric("No. of Products", f"{filtered_df['item_id'].nunique():,}")

st.divider()

# Revenue overview
st.subheader("Revenue Overview")

monthly_revenue = filtered_df.groupby("month_start", as_index=False)["revenue"].sum()

fig = px.line(
    monthly_revenue,
    x="month_start",
    y="revenue",
    markers=True,
    title="Revenue by Month"
)
fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Revenue"
)

rev_col1, rev_col2 = st.columns([2, 1])

with rev_col1:
    st.plotly_chart(fig, use_container_width=True)

with rev_col2:
    st.subheader("Current Month Target")

    full_monthly = (
        df.groupby("month_start", as_index=False)["revenue"]
        .sum()
        .sort_values("month_start")
    )

    full_monthly["t"] = np.arange(len(full_monthly))

    model = LinearRegression()
    model.fit(full_monthly[["t"]], full_monthly["revenue"])

    target_month = pd.Timestamp("2016-04-01")
    t_val = full_monthly.loc[
        full_monthly["month_start"] == target_month,
        "t"
    ].iloc[0]

    target = model.predict([[t_val]])[0]
    actual = df[df["month_start"] == target_month]["revenue"].sum()

    progress = actual / target if target > 0 else 0
    progress_bar_value = min(progress, 1)

    st.metric("Target Revenue", f"${target:,.0f}")
    st.metric("April 2016 Revenue", f"${actual:,.0f}")

    st.progress(progress_bar_value)
    st.caption(f"{progress * 100:.1f}% of target achieved")

    if progress >= 1:
        st.success("Target achieved.")
    else:
        st.info("Revenue is still below the target.")

month_count = (
    (end_month.year - start_month.year) * 12
    + (end_month.month - start_month.month)
    + 1
)

if month_count >= 12:
    yearly_revenue = filtered_df.groupby("year", as_index=False)["revenue"].sum()

    fig = px.bar(
        yearly_revenue,
        x="year",
        y="revenue",
        title="Revenue by Year"
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Sales trend
st.subheader("Sales Trend")

daily_sales = filtered_df.groupby("date", as_index=False)["sales"].sum()

fig = px.line(
    daily_sales,
    x="date",
    y="sales",
    title="Daily Sales Trend"
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Category and state
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    cat_sales = (
        filtered_df.groupby("cat_id", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=True)
    )
    bar_chart(cat_sales, "sales", "cat_id", "Sales by Category")

with col2:
    st.subheader("Sales Share by State")
    state_sales = filtered_df.groupby("state_id", as_index=False)["sales"].sum()

    fig = px.pie(
        state_sales,
        names="state_id",
        values="sales",
        title="Sales Distribution by State",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# Top stores and products side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Stores by Sales")
    top_bar(filtered_df, "store_id", "sales", "Top 10 Stores by Sales")

with col2:
    st.subheader("Top 10 Products by Sales")
    top_bar(filtered_df, "item_id", "sales", "Top 10 Products by Sales")

# Weekday analysis
st.subheader("Sales by Weekday")

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_sales = filtered_df.groupby("weekday", as_index=False)["sales"].mean()

weekday_sales["weekday"] = pd.Categorical(
    weekday_sales["weekday"],
    categories=weekday_order,
    ordered=True
)

weekday_sales = weekday_sales.sort_values("weekday")

fig = px.line(
    weekday_sales,
    x="weekday",
    y="sales",
    markers=True,
    title="Average Sales by Weekday"
)
fig.update_layout(
    xaxis_title="Weekday",
    yaxis_title="Average Sales"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Price sensitivity
st.subheader("Price Sensitivity")

st.write(
    "This estimates how sales may change when price changes using a simple log-log regression. "
    "It is useful for analysis, but it does not prove causation."
)

elasticity_base_df = filtered_df[filtered_df["sales"] > 0]

valid_products = (
    elasticity_base_df.groupby("item_id")["sell_price"]
    .nunique()
    .reset_index(name="num_price_points")
)

valid_products = valid_products[valid_products["num_price_points"] >= 5]

item_sales_rank = (
    elasticity_base_df[
        elasticity_base_df["item_id"].isin(valid_products["item_id"])
    ]
    .groupby("item_id", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

if item_sales_rank.empty:
    st.info("No products have enough price variation for price sensitivity analysis.")
else:
    selected_item = st.selectbox(
        "Select Product",
        item_sales_rank["item_id"].tolist()
    )

    item_df = elasticity_base_df[elasticity_base_df["item_id"] == selected_item]

    elasticity_df = (
        item_df.groupby("sell_price", as_index=False)
        .agg(
            avg_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            days_observed=("date", "nunique")
        )
    )

    X = np.log(elasticity_df[["sell_price"]])
    y = np.log(elasticity_df["avg_sales"])

    model = LinearRegression()
    model.fit(X, y)

    elasticity = model.coef_[0]
    elasticity_df["predicted_avg_sales"] = np.exp(model.predict(X))

    cols = st.columns(2)
    cols[0].metric("Estimated Elasticity", f"{elasticity:.3f}")
    cols[1].metric("No. of Price Points", len(elasticity_df))

    if elasticity < -1:
        st.info("This product appears price-sensitive.")
    elif elasticity < 0:
        st.info("This product appears weakly price-sensitive.")
    else:
        st.info("This product does not show the usual negative price-demand relationship.")

    fig = px.scatter(
        elasticity_df,
        x="sell_price",
        y="avg_sales",
        title=f"Price vs Average Sales for {selected_item}"
    )

    fig.add_scatter(
        x=elasticity_df["sell_price"],
        y=elasticity_df["predicted_avg_sales"],
        mode="lines+markers",
        name="Regression Fit"
    )

    fig.update_layout(
        xaxis_title="Sell Price",
        yaxis_title="Average Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Method used: log(avg_sales) = a + b log(price). "
        "Only products with at least three price points are selectable. "
        "Only positive-sales observations are used. This does not prove causation."
    )

# Data preview
st.subheader("Data Preview")
st.dataframe(filtered_df.head(1000))