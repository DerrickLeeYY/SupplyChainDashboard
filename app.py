import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")

DATA_FILE = "cleaned_m5_dashboard_data.parquet"


@st.cache_data
def load_filter_data():
    df = pd.read_parquet(
        DATA_FILE,
        columns=["date", "state_id", "store_id", "cat_id"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["month_start"] = df["date"].values.astype("datetime64[M]")

    return df


@st.cache_data
def get_base_outputs(start_month, end_month, state_filter, store_filter, cat_filter):
    df = pd.read_parquet(
        DATA_FILE,
        columns=["date", "state_id", "store_id", "cat_id", "sales", "revenue", "weekday"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("int16")
    df["revenue"] = df["revenue"].astype("float32")
    df["year"] = df["date"].dt.year
    df["month_start"] = df["date"].values.astype("datetime64[M]")

    start_month = pd.to_datetime(start_month)
    end_month = pd.to_datetime(end_month)

    df = df[
        (df["date"] >= start_month) &
        (df["date"] <= end_month) &
        (df["state_id"].isin(state_filter)) &
        (df["store_id"].isin(store_filter)) &
        (df["cat_id"].isin(cat_filter))
    ]

    if df.empty:
        return None

    kpis = {
        "total_sales": df["sales"].sum(),
        "total_revenue": df["revenue"].sum(),
        "avg_daily_sales": df.groupby("date")["sales"].sum().mean(),
    }

    monthly_revenue = df.groupby("month_start", as_index=False)["revenue"].sum()
    yearly_revenue = df.groupby("year", as_index=False)["revenue"].sum()
    daily_sales = df.groupby("date", as_index=False)["sales"].sum()

    cat_sales = (
        df.groupby("cat_id", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=True)
    )

    state_sales = df.groupby("state_id", as_index=False)["sales"].sum()

    store_sales = (
        df.groupby("store_id", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(10)
        .sort_values("sales", ascending=True)
    )

    weekday_sales = df.groupby("weekday", as_index=False)["sales"].mean()

    return kpis, monthly_revenue, yearly_revenue, daily_sales, cat_sales, state_sales, store_sales, weekday_sales


@st.cache_data
def get_target_outputs():
    df = pd.read_parquet(DATA_FILE, columns=["date", "revenue"])
    df["date"] = pd.to_datetime(df["date"])
    df["month_start"] = df["date"].values.astype("datetime64[M]")
    df["revenue"] = df["revenue"].astype("float32")

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

    return target, actual


@st.cache_data
def get_product_sales(start_month, end_month, state_filter, store_filter, cat_filter):
    df = pd.read_parquet(
        DATA_FILE,
        columns=["date", "state_id", "store_id", "cat_id", "item_id", "sales"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("int16")

    start_month = pd.to_datetime(start_month)
    end_month = pd.to_datetime(end_month)

    df = df[
        (df["date"] >= start_month) &
        (df["date"] <= end_month) &
        (df["state_id"].isin(state_filter)) &
        (df["store_id"].isin(store_filter)) &
        (df["cat_id"].isin(cat_filter))
    ]

    num_products = df["item_id"].nunique()

    top_products = (
        df.groupby("item_id", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(10)
        .sort_values("sales", ascending=True)
    )

    return num_products, top_products


@st.cache_data
def get_price_sensitivity_data(start_month, end_month, state_filter, store_filter, cat_filter):
    df = pd.read_parquet(
        DATA_FILE,
        columns=["date", "state_id", "store_id", "cat_id", "item_id", "sales", "sell_price"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("int16")
    df["sell_price"] = df["sell_price"].astype("float32")

    start_month = pd.to_datetime(start_month)
    end_month = pd.to_datetime(end_month)

    df = df[
        (df["date"] >= start_month) &
        (df["date"] <= end_month) &
        (df["state_id"].isin(state_filter)) &
        (df["store_id"].isin(store_filter)) &
        (df["cat_id"].isin(cat_filter)) &
        (df["sales"] > 0)
    ]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    price_summary = (
        df.groupby(["item_id", "sell_price"], as_index=False)
        .agg(
            avg_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            days_observed=("date", "nunique")
        )
    )

    valid_products = (
        price_summary.groupby("item_id")["sell_price"]
        .nunique()
        .reset_index(name="num_price_points")
    )

    valid_products = valid_products[valid_products["num_price_points"] >= 5]

    item_sales_rank = (
        df[df["item_id"].isin(valid_products["item_id"])]
        .groupby("item_id", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    return item_sales_rank, price_summary


def bar_chart(data, x, y, title):
    fig = px.bar(data, x=x, y=y, orientation="h", title=title)
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title()
    )
    st.plotly_chart(fig, use_container_width=True)


filter_df = load_filter_data()

st.title("MA6721 Group 1 CA3 M5 Sales Dashboard")
st.write(
    "This dashboard uses a snapshot of the data from 1st February 2011 to 24th April 2016. "
    "It treats today as 24th April 2016."
)

# Sidebar filters
st.sidebar.header("Filters")

month_options = sorted(filter_df["month_start"].unique())
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
    sorted(filter_df["state_id"].unique()),
    default=sorted(filter_df["state_id"].unique())
)

store_filter = st.sidebar.multiselect(
    "Select Store",
    sorted(filter_df["store_id"].unique()),
    default=sorted(filter_df["store_id"].unique())
)

cat_filter = st.sidebar.multiselect(
    "Select Category",
    sorted(filter_df["cat_id"].unique()),
    default=sorted(filter_df["cat_id"].unique())
)

state_filter = tuple(state_filter)
store_filter = tuple(store_filter)
cat_filter = tuple(cat_filter)

outputs = get_base_outputs(
    start_month,
    end_month,
    state_filter,
    store_filter,
    cat_filter
)

if outputs is None:
    st.warning("No data available for the selected filters.")
    st.stop()

(
    kpis,
    monthly_revenue,
    yearly_revenue,
    daily_sales,
    cat_sales,
    state_sales,
    store_sales,
    weekday_sales
) = outputs

num_products, top_products = get_product_sales(
    start_month,
    end_month,
    state_filter,
    store_filter,
    cat_filter
)

# KPIs
cols = st.columns(4)
cols[0].metric("Total Sales", f"{kpis['total_sales']:,.0f}")
cols[1].metric("Total Revenue", f"${kpis['total_revenue']:,.2f}")
cols[2].metric("Avg Daily Sales", f"{kpis['avg_daily_sales']:,.2f}")
cols[3].metric("No. of Products", f"{num_products:,}")

st.divider()

# Revenue overview
st.subheader("Revenue Overview")

fig = px.line(
    monthly_revenue,
    x="month_start",
    y="revenue",
    markers=True,
    title="Revenue by Month"
)
fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")

rev_col1, rev_col2 = st.columns([2, 1])

with rev_col1:
    st.plotly_chart(fig, use_container_width=True)

with rev_col2:
    st.subheader("Current Month Target")

    target, actual = get_target_outputs()

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
    fig = px.bar(
        yearly_revenue,
        x="year",
        y="revenue",
        title="Revenue by Year"
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Sales trend
st.subheader("Sales Trend")

fig = px.line(
    daily_sales,
    x="date",
    y="sales",
    title="Daily Sales Trend"
)
fig.update_layout(xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Category and state
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    bar_chart(cat_sales, "sales", "cat_id", "Sales by Category")

with col2:
    st.subheader("Sales Share by State")
    fig = px.pie(
        state_sales,
        names="state_id",
        values="sales",
        title="Sales Distribution by State",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# Top stores and products
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Stores by Sales")
    bar_chart(store_sales, "sales", "store_id", "Top 10 Stores by Sales")

with col2:
    st.subheader("Top 10 Products by Sales")
    bar_chart(top_products, "sales", "item_id", "Top 10 Products by Sales")

# Weekday analysis
st.subheader("Sales by Weekday")

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

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
fig.update_layout(xaxis_title="Weekday", yaxis_title="Average Sales")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Price sensitivity
st.subheader("Price Sensitivity")

st.write(
    "This estimates how sales may change when price changes using a simple log-log regression. "
    "It is useful for analysis, but it does not prove causation."
)

item_sales_rank, price_summary = get_price_sensitivity_data(
    start_month,
    end_month,
    state_filter,
    store_filter,
    cat_filter
)

if item_sales_rank.empty:
    st.info("No products have enough price variation for price sensitivity analysis.")
else:
    selected_item = st.selectbox(
        "Select Product",
        item_sales_rank["item_id"].tolist()
    )

    elasticity_df = price_summary[price_summary["item_id"] == selected_item].copy()

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
        "Only products with at least five price points are selectable. "
        "Only positive-sales observations are used. This does not prove causation."
    )