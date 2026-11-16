
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="Amazon Product Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "amazon.csv"
PROFILE_PATH = BASE_DIR / "product_segment_profile.csv"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


st.title("Amazon Product Analytics Dashboard")
st.caption("Data Analytics with AI Integration | Internship Project")

# Load dataset
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load amazon.csv: {e}")
    st.stop()


# Sidebar navigation
page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Project Overview",
        "Task 1 — Product Segmentation",
        "Task 2 — Discount & Rating",
        "Task 3 — Sentiment Analysis",
        "Task 4 — Product Ranking",
        "Task 5 — Anomaly Detection",
        "Task 6 — Rating Prediction",
    ]
)


# PROJECT OVERVIEW

if page == "Project Overview":
    st.header("Project Overview")

    st.markdown(
        """
        This project analyzes Amazon product data using data analytics,
        statistical methods, clustering, sentiment analysis, ranking,
        anomaly detection, and predictive modeling.
        """
    )

    # Identify the top-level category from the original category path
    if "category" in df.columns:
        main_categories = (
            df["category"]
            .dropna()
            .astype(str)
            .str.split("|", regex=False)
            .str[0]
        )
        category_count = main_categories.nunique()
    else:
        category_count = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Product Records", f"{len(df):,}")
    col2.metric("Dataset Features", f"{len(df.columns):,}")
    col3.metric("Top-Level Categories", f"{category_count:,}")

    st.divider()

    st.subheader("Explore the Internship Tasks")

    task_col1, task_col2, task_col3 = st.columns(3)

    with task_col1:
        st.markdown("**Task 1 — Product Segmentation**")
        st.caption("Discover product groups and segment profiles.")

        st.markdown("**Task 2 — Discount & Rating**")
        st.caption("Examine correlations and regression results.")

    with task_col2:
        st.markdown("**Task 3 — Sentiment Analysis**")
        st.caption("Analyze review sentiment and category differences.")

        st.markdown("**Task 4 — Product Ranking**")
        st.caption("Compare products using a weighted ranking model.")

    with task_col3:
        st.markdown("**Task 5 — Anomaly Detection**")
        st.caption("Investigate unusual product records.")

        st.markdown("**Task 6 — Rating Prediction**")
        st.caption("Evaluate regression models and feature importance.")

    st.divider()

    st.subheader("Dataset Preview")

    preview_columns = [
        "product_id",
        "product_name",
        "category",
        "discounted_price",
        "rating",
        "rating_count",
    ]

    available_columns = [
        col for col in preview_columns if col in df.columns
    ]

    preview_df = df[available_columns].head(10).copy()

    if "category" in preview_df.columns:
        preview_df["category"] = (
            preview_df["category"]
            .fillna("Unknown")
            .astype(str)
            .str.split("|", regex=False)
            .str[0]
        )

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View All Dataset Columns"):
        st.write(df.columns.tolist())


# TASK 1 — PRODUCT SEGMENTATION

# TASK 1 — PRODUCT SEGMENTATION
elif page == "Task 1 — Product Segmentation":
    st.header("Task 1: Product Segmentation")

    st.write(
        "Products were grouped using price, discount percentage, "
        "rating, and rating-count features. Five clusters were "
        "selected using the elbow method and silhouette score."
    )

    if not PROFILE_PATH.exists():
        st.warning(
            "product_segment_profile.csv was not found. "
            "Ensure it is in the same folder as app.py."
        )
    else:
        profile = load_data(PROFILE_PATH)

        # Remove an exported index column if present
        profile = profile.loc[
            :, ~profile.columns.astype(str).str.startswith("Unnamed:")
        ]

        # Exclude median price and median discount from the dashboard table
        display_profile = profile.drop(
            columns=["median_price", "median_discount", "median_rating", "median_rating_count"],
            errors="ignore"
        )

        st.subheader("Segment Profiles")
        st.dataframe(display_profile, use_container_width=True)

        metric_labels = {
            "product_count": "Number of Products",
            "avg_price": "Average Price (₹)",
            "avg_discount": "Average Discount (%)",
            "avg_rating": "Average Rating",
            "median_rating": "Median Rating",
            "avg_rating_count": "Average Rating Count",
            "median_rating_count": "Median Rating Count",
            "segment_percentage": "Segment Size (%)",
        }

        available_metrics = [
            col for col in metric_labels
            if col in profile.columns
        ]

        if available_metrics:
            st.subheader("Compare Segment Metrics")

            default_metric = (
                "avg_rating"
                if "avg_rating" in available_metrics
                else available_metrics[0]
            )

            metric = st.selectbox(
                "Choose a metric",
                available_metrics,
                index=available_metrics.index(default_metric),
                format_func=lambda x: metric_labels[x]
            )

            cluster_col = (
                "cluster"
                if "cluster" in profile.columns
                else profile.columns[0]
            )

            fig = px.bar(
                profile,
                x=cluster_col,
                y=metric,
                title=f"{metric_labels[metric]} by Cluster",
                text_auto=".2s",
                labels={
                    cluster_col: "Product Segment",
                    metric: metric_labels[metric]
                }
            )

            fig.update_layout(
                xaxis_title="Product Segment",
                yaxis_title=metric_labels[metric]
            )

            st.plotly_chart(fig, use_container_width=True)

            if metric == "segment_percentage":
                st.caption(
                    "Shows the percentage of products in each segment."
                )
            elif metric in ["avg_rating", "median_rating"]:
                st.caption(
                    "Compare product ratings across the five segments."
                )
            elif metric in ["avg_price", "avg_rating_count", "median_rating_count"]:
                st.caption(
                    "Compare price or rating-count levels across segments."
                )
        else:
            st.warning(
                "No recognized segment metrics were found in the CSV."
            )



# TASK 2 — DISCOUNT AND RATING


elif page == "Task 2 — Discount & Rating":
    st.header("Task 2: Discount–Rating Relationship")

    st.write(
        "This analysis examines the relationship between product "
        "discounts and ratings. The results show association, "
        "not causation."
    )

    # Correlation results from the completed analysis
    c1, c2 = st.columns(2)

    c1.metric("Pearson Correlation", "-0.1557")
    c1.caption("P-value: 2.1774e-09")

    c2.metric("Spearman Correlation", "-0.1462")
    c2.caption("P-value: 1.9623e-08")

    st.subheader("Correlation Interpretation")
    st.write(
        "Both correlations indicate a weak negative association "
        "between discount percentage and product rating. "
        "Both are statistically significant, but correlation "
        "does not establish causation."
    )

    st.subheader("Average Rating by Discount Range")

    discount_data = pd.DataFrame({
        "Discount Range": [
            "0–20%", "21–40%", "41–60%", "61–80%", "81–100%"
        ],
        "Average Rating": [4.173, 4.126, 4.079, 4.076, 3.976]
    })

    fig = px.bar(
        discount_data,
        x="Discount Range",
        y="Average Rating",
        title="Average Product Rating by Discount Range",
        text="Average Rating",
        labels={
            "Discount Range": "Discount Range",
            "Average Rating": "Average Rating"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_yaxes(
        range=[0, 5],
        title="Average Rating (out of 5)"
    )

    fig.update_layout(
        xaxis_title="Discount Range",
        uniformtext_minsize=10,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regression Results")

    regression_data = pd.DataFrame({
        "Analysis": ["Full Dataset", "After Outlier Filtering"],
        "Discount Coefficient": [-0.001287, -0.001959],
        "P-value": [0.0011995, 0.00001532]
    })

    st.dataframe(
        regression_data.style.format({
            "Discount Coefficient": "{:.6f}",
            "P-value": "{:.6g}"
        }),
        use_container_width=True
    )

    st.info(
        "The discount coefficient is negative and statistically "
        "significant in both models. However, the association is "
        "weak and does not establish that discounts cause lower ratings."
    )

    st.caption(
        "Note: The 81–100% discount group contains only 59 products."
    )




# TASK 3 — SENTIMENT ANALYSIS

elif page == "Task 3 — Sentiment Analysis":
    st.header("Task 3: Review-Based Sentiment Analysis")

    st.write(
        "VADER sentiment analysis was applied to combined review "
        "titles and contents. Sentiment is measured at the "
        "product-record level."
    )

    # Results from the completed Task 3 analysis
    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [1410, 53, 2]
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", "1,410")
    c2.metric("Negative", "53")
    c3.metric("Neutral", "2")
    st.subheader("Sentiment Distribution")

    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Count",
        title="Distribution of Review Sentiment",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Negative Sentiment by Major Category")

    category_data = pd.DataFrame({
        "Category": [
            "Computers & Accessories",
            "Electronics",
            "Home & Kitchen"
        ],
        "Total Records": [453, 526, 448],
        "Negative Reviews": [12, 21, 19],
        "Negative Percentage": [
            12 / 453 * 100,
            21 / 526 * 100,
            19 / 448 * 100
        ]
    })

    st.dataframe(
        category_data.style.format({
            "Negative Percentage": "{:.2f}%"
        }),
        use_container_width=True
    )

    fig = px.bar(
        category_data,
        x="Category",
        y="Negative Percentage",
        title="Negative Sentiment Percentage by Category",
        text_auto=".2f",
        labels={
            "Category": "Product Category",
            "Negative Percentage": "Negative Sentiment (%)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Test")

    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-square Statistic", "1.914")
    c2.metric("Degrees of Freedom", "2")
    c3.metric("P-value", "0.384053")

    st.info(
        "There is insufficient evidence that negative sentiment "
        "proportions differ across the three major product categories "
        "at the 5% significance level."
    )

    st.caption(
        "VADER is a rule-based sentiment tool. Negative sentiment "
        "does not independently prove a product defect."
    )



# TASK 4 — PRODUCT RANKING

elif page == "Task 4 — Product Ranking":
    st.header("Task 4: Product Ranking Model")

    st.write(
        "Products were ranked using Bayesian-adjusted ratings, "
        "review popularity, discount percentage, and affordability."
    )

    st.subheader("Ranking Method")

    weights = pd.DataFrame({
        "Ranking Component": [
            "Bayesian-adjusted rating",
            "Popularity (rating count)",
            "Discount percentage",
            "Affordability"
        ],
        "Weight": [0.50, 0.20, 0.20, 0.10]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(weights, use_container_width=True, hide_index=True)

    with col2:
        fig = px.pie(
            weights,
            names="Ranking Component",
            values="Weight",
            title="Ranking Score Weights",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Ranked Products")

    top_products = pd.DataFrame({
        "Rank": range(1, 11),
        "Product ID": [
            "B014I8SX4Y", "B00NH11PEY", "B00NH11KIK",
            "B07Q4QV1DL", "B088ZFJY82", "B07KSMBL2H",
            "B095RTJH1M", "B00NH13Q8W", "B0BR4F878Q",
            "B0B23LW7NV"
        ],
        "Product Name": [
            "Amazon Basics High-Speed HDMI Cable, 6 Feet (2-Pack)",
            "AmazonBasics USB 2.0 Extension Cable, 9.8 Feet",
            "AmazonBasics USB 2.0 Cable, 6 Feet",
            "ELV Adjustable Mobile Phone Stand, Black",
            "ELV Adjustable Mobile Phone Stand, Silver",
            "AmazonBasics HDMI Cable, 3 Feet",
            "Spigen EZ Fit Screen Protector for iPhone 14/13 Pro",
            "AmazonBasics USB 2.0 Extension Cable, 2-Pack",
            "Swiffer Instant Electric Water Heater Faucet",
            "Spigen EZ Fit Screen Protector for iPhone 14 Pro Max"
        ],
        "Rating": [4.4, 4.5, 4.5, 4.5, 4.5, 4.4, 4.6, 4.5, 4.8, 4.7],
        "Rating Count": [
            426973, 74976, 107687, 28978, 28978,
            426973, 26603, 74977, 53803, 7779
        ],
        "Discounted Price (₹)": [
            309, 199, 209, 269, 314, 219, 999, 299, 1439, 999
        ],
        "Discount (%)": [78, 73, 70, 82, 79, 69, 66, 63, 28, 66],
        "Ranking Score": [
            0.806, 0.804, 0.804, 0.800, 0.792,
            0.791, 0.780, 0.777, 0.777, 0.775
        ]
    })

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        top_products,
        x="Rank",
        y="Ranking Score",
        hover_name="Product Name",
        title="Top 10 Products by Ranking Score",
        text_auto=".3f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranking Stability")

    col1, col2 = st.columns(2)
    col1.metric("Top-10 Products in Common", "10 / 10")
    col2.metric("Top-10 Overlap", "100%")

    st.info(
        "All ten products remained in the top ten under the tested "
        "alternative weighting scheme. This indicates stability "
        "under that particular comparison, not every possible weighting."
    )

    st.caption(
        "The ranking is a weighted analytical score, not a verified "
        "measure of profitability or overall product quality."
    )



# TASK 5 — ANOMALY DETECTION

elif page == "Task 5 — Anomaly Detection":
    st.header("Task 5: Anomaly Detection")

    st.write(
        "This analysis identifies unusual product records using "
        "the IQR method and Isolation Forest. An anomaly is not "
        "automatically an error or fraudulent listing."
    )

    anomaly_path = BASE_DIR / "task5_anomaly_investigation.csv"

    if not anomaly_path.exists():
        st.warning(
            "task5_anomaly_investigation.csv was not found. "
            "Ensure it is in the same folder as app.py."
        )
    else:
        anomalies = load_data(anomaly_path)

        # Summary based on the investigation file
        col1, col2, col3 = st.columns(3)

        col1.metric("Products Investigated", len(anomalies))

        if "iqr_anomaly" in anomalies.columns:
            col2.metric(
                "IQR Flags in This Table",
                int(anomalies["iqr_anomaly"].astype(bool).sum())
            )

        if "isolation_anomaly" in anomalies.columns:
            col3.metric(
                "Isolation Forest Flags in This Table",
                int(anomalies["isolation_anomaly"].astype(bool).sum())
            )

        st.subheader("Anomaly Investigation Table")

        st.dataframe(
            anomalies,
            use_container_width=True,
            hide_index=True
        )

        if {
            "discounted_price",
            "rating",
            "anomaly_score"
        }.issubset(anomalies.columns):
            st.subheader("Anomaly Score vs Product Rating")

            fig = px.scatter(
                anomalies,
                x="rating",
                y="anomaly_score",
                size="discounted_price",
                color="isolation_anomaly"
                if "isolation_anomaly" in anomalies.columns
                else None,
                hover_name="product_name"
                if "product_name" in anomalies.columns
                else None,
                title="Investigated Products by Rating and Anomaly Score",
                labels={
                    "rating": "Product Rating",
                    "anomaly_score": "Anomaly Score",
                    "discounted_price": "Discounted Price (₹)",
                    "isolation_anomaly": "Isolation Forest Flag"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This page displays the saved investigation subset, "
            "not every anomalous record found in the complete dataset."
        )


# TASK 6 — RATING PREDICTION

elif page == "Task 6 — Rating Prediction":
    st.header("Task 6: Predictive Rating Analysis")

    st.write(
        "Two regression models were evaluated to predict product ratings "
        "using price, discount, rating count, and product category."
    )

    st.subheader("Model Performance")

    model_results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "CV MAE": [0.2039, 0.1847],
        "CV RMSE": [0.2789, 0.2658],
        "CV R²": [0.0959, 0.1765],
        "Test MAE": [0.1978, 0.1847],
        "Test RMSE": [0.2607, 0.2648],
        "Test R²": [0.1243, 0.0965]
    })

    st.dataframe(
        model_results.style.format({
            "CV MAE": "{:.4f}",
            "CV RMSE": "{:.4f}",
            "CV R²": "{:.4f}",
            "Test MAE": "{:.4f}",
            "Test RMSE": "{:.4f}",
            "Test R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Compare Test-Set Performance")

    performance_metric = st.selectbox(
        "Choose evaluation metric",
        ["Test MAE", "Test RMSE", "Test R²"]
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=performance_metric,
        color="Model",
        title=f"Model Comparison — {performance_metric}",
        text_auto=".4f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Random Forest Feature Importance")

    importance = pd.DataFrame({
        "Feature": [
            "main_category",
            "rating_count",
            "discount_percentage",
            "discounted_price",
            "actual_price"
        ],
        "Importance Mean": [
            0.02636, 0.01935, 0.01466, 0.01227, 0.01060
        ],
        "Importance Std": [
            0.00429, 0.00511, 0.00327, 0.00242, 0.00198
        ]
    })

    fig = px.bar(
        importance.sort_values("Importance Mean"),
        x="Importance Mean",
        y="Feature",
        error_x="Importance Std",
        orientation="h",
        title="Random Forest Permutation Importance",
        labels={
            "Importance Mean": "Increase in MAE when shuffled",
            "Feature": "Feature"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Random Forest achieved better cross-validation MAE and R², "
        "while Linear Regression achieved slightly better test RMSE "
        "and R². Neither model explains most of the variation in ratings, "
        "so predictive performance is modest."
    )

    st.caption(
        "Permutation importance describes how much prediction error "
        "increases when a feature is shuffled. It does not establish causation."
    )
