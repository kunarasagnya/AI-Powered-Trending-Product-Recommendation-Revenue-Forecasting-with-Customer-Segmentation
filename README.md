# AI-Powered Trending Product Recommendation and Revenue Forecasting with Customer Segmentation

## Overview
In today’s competitive e-commerce world, companies face difficulties in identifying which products are likely to trend, how much revenue they can generate, and which customer segments are most likely to purchase them.  
With thousands of new products listed every day, manually analyzing sales data, reviews, ratings, and prices becomes extremely time-consuming.

This project aims to solve these challenges by building an AI-based system that predicts trending products, forecasts revenue, and segments customers based on similar purchase behavior.  
The system is supported by an interactive Streamlit dashboard that visualizes all insights clearly and helps companies make data-driven business decisions.

---

## Problem Statement
E-commerce businesses often struggle with questions like:
- Which products are likely to become bestsellers in the coming months?
- How much revenue can each product generate?
- Which customer groups prefer which products?

Due to the large volume of data, traditional manual analysis is not enough. As a result, businesses face problems such as:
- Launching products that fail in the market
- Missing out on promoting high-demand items
- Ineffective stock and marketing strategies

To overcome these challenges, this project develops an intelligent solution using machine learning and data analysis.

---

## Project Objectives
The main objectives of this project are:
1. Predict trending products that may perform well in the future.
2. Forecast the expected monthly revenue for each product.
3. Segment customers based on their purchasing patterns using clustering techniques.
4. Recommend top-performing products and categories to businesses.
5. Build an interactive dashboard that combines all these insights in one place.

---

## Dataset Description
The dataset used in this project is the **Amazon Product Sales Dataset (2025)** containing data of more than 42,000 electronic products.  
It includes details such as:
- Product titles, categories, and ratings  
- Number of reviews and total revenue  
- Discount percentages and price details  
- Monthly purchases and other sales indicators  

The dataset was cleaned and preprocessed by handling missing values, removing duplicates, and standardizing columns.  
Encoding `ISO-8859-1` was used to correctly read product titles that contain special symbols or currency characters.

---

## Project Workflow
1. **Data Collection and Cleaning:**  
   The raw dataset was imported and cleaned by removing duplicates, handling missing values, and ensuring correct data types.

2. **Exploratory Data Analysis (EDA):**  
   Various visualizations such as histograms, scatter plots, and boxplots were created to understand relationships between features like ratings, price, and revenue.

3. **Feature Engineering:**  
   New features such as discount ratio, log of total reviews, and total revenue were added to improve the model’s performance.

4. **Revenue Forecasting (XGBoost):**  
   A regression model was trained using XGBoost to predict the monthly revenue for products based on ratings, reviews, and discounts.

5. **Customer Segmentation (K-Means):**  
   Products were grouped into clusters based on features like number of reviews, recent purchases, discounted price, and rating.

6. **Trending Product Recommendation:**  
   Top products were identified based on total revenue and visualized using bar charts.

7. **Dashboard Development:**  
   A Streamlit dashboard was built to combine all insights and allow users to interact with data easily.

---

## Dashboard Features

### Data Overview
Displays a summary of the dataset including total products, average rating, total revenue, and basic statistics.

### Exploratory Data Analysis
Shows visual patterns such as distribution of ratings, revenue by category, and relationships between discounts and purchases.

### Revenue Forecasting
Predicts the estimated monthly revenue for a given product based on user input for discount, rating, and reviews.

### Trending Product Recommendations
Lists top 10 products based on total revenue with visualization for quick comparison.

### Customer Segmentation
Uses K-Means clustering to group similar products or customers and shows them using scatter plots for easy understanding.

---

## Expected Outcomes
- Ability to forecast product revenue with good accuracy.  
- Identification of high-demand and trending products before they peak.  
- Segmentation of customers or products for personalized targeting.  
- Easy visualization of insights for business decision-making.  
- A reusable AI pipeline that can be applied to future datasets.

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost  
- **Dashboard Framework:** Streamlit  
- **Model Serialization:** Joblib  
- **Development Tools:** Jupyter Notebook, VS Code  
- **Version Control:** GitHub  

---

## Insights and Learnings
From this project, several useful insights were observed:
- Discounts strongly influence sales, but excessive discounts may lower product ratings.  
- Certain product categories consistently generate higher revenue.  
- Products with more reviews and higher ratings tend to perform better financially.  
- Customer segments show clear differences in buying behavior — some focus on price, others on brand or quality.  
- Interactive dashboards make it easy for non-technical users to understand and act on data insights.

---

## Project Scope
This project focuses on analyzing, predicting, and recommending Amazon products using data-driven methods.  
It helps e-commerce companies:
- Predict revenue trends and upcoming bestsellers  
- Improve marketing and stock planning decisions  
- Understand different types of customers or product clusters  
- Make better business decisions through visual analytics  

---

## Limitations
- Dataset is limited to a specific category (electronics).  
- Predictions are based on past data; real-world factors may change outcomes.  
- Some external factors like promotions, holidays, or competitor actions are not included.  
- The model requires retraining periodically as new data becomes available.  
- High computation time may be required for very large datasets.

---

## How to Run the Project

1. Clone the repository  

2. Navigate to the project folder  

3. Install all dependencies  

4. Run the Streamlit application  

---

## Conclusion
This project provides a complete end-to-end solution that can help e-commerce businesses identify trending products, predict their potential revenue, and group products or customers for better marketing and decision-making.  
The dashboard makes it easy for any user — even those without technical knowledge — to visualize and understand insights directly from the data.

---

## Developed By
**Rasagnya Kuna**  
M.Sc. Data Science – University Post Graduate College(O.U.),Secunderabad
Focus Area: Artificial Intelligence, Machine Learning, and Business Data Analysis
