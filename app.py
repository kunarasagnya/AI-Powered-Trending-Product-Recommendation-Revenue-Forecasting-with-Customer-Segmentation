import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------
# 1️ Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Amazon Sales Dashboard", layout="wide")

# ----------------------------------------------------------
# 2️ Load Data and Model
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/amazon_products_sales_data_cleaned.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/xgb_model.pkl")
    return model

df = load_data()
xgb_model = load_model()

# ----------------------------------------------------------
# 3️ Sidebar Navigation
# ----------------------------------------------------------
st.sidebar.title("Amazon Sales Intelligence Dashboard")
menu = st.sidebar.radio(
    "Select Section:",
    [
        "Data Overview",
        "Exploratory Data Analysis",
        "Revenue Forecasting",
        "Trending Products Recommendation",
        "Customer Segmentation"
    ]
)

# ----------------------------------------------------------
# 4️ Data Overview
# ----------------------------------------------------------
if menu == "Data Overview":
    st.title("Amazon Product Sales Data Overview")
    st.write("Explore the structure and summary of the cleaned Amazon sales dataset.")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(df))
    col2.metric("Average Rating", round(df['product_rating'].mean(), 2))
    col3.metric("Total Revenue ($)", f"{df['revenue'].sum():,.2f}")

    st.subheader("Null Value Check")
    st.write(df.isnull().sum())

    st.subheader("Dataset Info")
    st.write(df.describe())

# ----------------------------------------------------------
# 5️ Exploratory Data Analysis
# ----------------------------------------------------------
elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Visual insights about key patterns in product ratings, prices, and sales.")

    st.subheader("Distribution of Ratings")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["product_rating"], kde=True, bins=30, color="skyblue", ax=ax)
    st.pyplot(fig)

    # Remove invalid/unknown categories
    if "product_category" in df.columns:
        valid_df = df[~df["product_category"].isin(["Not Available", "Unknown", "Unavailable"])]
        category_revenue = valid_df.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    else:
        category_cols = [col for col in df.columns if col.startswith("product_category_")]
        if category_cols:
            category_revenue = pd.Series({
                col.replace("product_category_", ""): df.loc[df[col] == 1, "revenue"].sum()
                for col in category_cols
            }).sort_values(ascending=False)
        else:
            st.warning("No category information available.")
            category_revenue = pd.Series(dtype=float)

    if not category_revenue.empty:
        st.subheader("Top Product Categories by Revenue")
        st.bar_chart(category_revenue)

    # Discount vs Purchases
    st.subheader("Discount vs Purchases")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x='discount_percentage', y='purchased_last_month', alpha=0.6, ax=ax)
    st.pyplot(fig)

    # Price Distribution
    st.subheader("Price Distribution by Category")
    if "product_category" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='product_category', y='discounted_price', data=valid_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Automated insights
    st.subheader("Automated Insights Summary")
    discount_corr = df["discount_percentage"].corr(df["purchased_last_month"])
    rating_corr = df["product_rating"].corr(df["revenue"])
    avg_rating = df["product_rating"].mean()
    top_cat = category_revenue.index[0] if not category_revenue.empty else "N/A"

    st.write(f"• Products with higher discounts tend to have higher sales volume (correlation = {discount_corr:.2f}).")
    st.write(f"• The category **{top_cat}** generates the highest total revenue.")
    st.write(f"• Higher ratings slightly correlate with better revenue (correlation = {rating_corr:.2f}).")
    st.write(f"• The average product rating is **{avg_rating:.2f}**, showing strong customer satisfaction overall.")

# ----------------------------------------------------------
# 6️ Revenue Forecasting
# ----------------------------------------------------------
elif menu == "Revenue Forecasting":
    st.title("🔮 Revenue Forecasting Using XGBoost Model")
    st.write("Predict potential revenue for a specific product based on its attributes.")

    # 🏷️ Product name input
    product_name = st.text_input("Enter Product Name:", placeholder="e.g., Apple iPhone 15")

    # User inputs
    discount = st.slider("Discount Percentage (%)", 0, 100, 20)
    rating = st.slider("Product Rating", 1.0, 5.0, 4.0, step=0.1)
    reviews = st.number_input("Total Reviews", min_value=0, value=100)

    # Retrieve model features
    model_features = xgb_model.get_booster().feature_names
    input_data = pd.DataFrame({f: [0] for f in model_features})

    # Update user-driven features
    if "discount_percentage" in input_data.columns:
        input_data["discount_percentage"] = discount
    if "product_rating" in input_data.columns:
        input_data["product_rating"] = rating
    if "total_reviews" in input_data.columns:
        input_data["total_reviews"] = reviews
    if "discount_ratio" in input_data.columns:
        input_data["discount_ratio"] = discount / 100
    if "log_total_reviews" in input_data.columns:
        input_data["log_total_reviews"] = np.log1p(reviews)

    # Fill missing numerical features with averages
    for col in input_data.columns:
        if col in df.columns and input_data[col].iloc[0] == 0:
            if np.issubdtype(df[col].dtype, np.number):
                input_data[col] = df[col].mean()

    # Prediction
    try:
        prediction = xgb_model.predict(input_data)[0]
        prediction = max(0, prediction)  # avoid negative revenue
        if product_name:
            st.success(f"Predicted Monthly Revenue for **{product_name}**: ${prediction:,.2f}")
        else:
            st.success(f"Predicted Monthly Revenue: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------------------------------
# 7️ Trending Products Recommendation
# ----------------------------------------------------------
elif menu == "Trending Products Recommendation":
    st.title("Trending Product Recommendations")
    st.write("Identify top-performing products based on recent purchases and predicted revenue.")

    if 'revenue' in df.columns:
        top_products = df.nlargest(10, 'revenue')[["product_title", "revenue", "product_rating", "discount_percentage"]]
        st.subheader("Top 10 Trending Products by Revenue")
        st.dataframe(top_products)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_products, x="revenue", y="product_title", palette="viridis", ax=ax)
        plt.title("Top 10 Products by Revenue")
        st.pyplot(fig)
    else:
        st.warning("Revenue column not found in dataset.")

# ----------------------------------------------------------
# 8️ Customer Segmentation (Clustering)
# ----------------------------------------------------------
elif menu == "Customer Segmentation":
    st.title("Customer/Product Segmentation")
    st.write("Cluster products based on similar sales, reviews, and price behaviors.")

    cluster_features = ['total_reviews', 'purchased_last_month', 'discounted_price', 'product_rating']
    X_cluster = df[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Cluster Summary (Mean Values)")
    cluster_summary = df.groupby('cluster')[cluster_features].mean()
    st.dataframe(cluster_summary)

    st.subheader("Visualizing Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='purchased_last_month', y='discounted_price', hue='cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.success("Clustering completed. Products are grouped into segments based on their characteristics.")
