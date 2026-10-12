import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Classification",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏠 House Price Classification")
st.write(
    "Compare Logistic Regression, KNN, Decision Tree "
    "and Random Forest."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("house_prices_practice.csv")

    return df


df = load_data()


# =========================================================
# FEATURES
# =========================================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "LotArea"
]


# =========================================================
# CREATE TARGET
# =========================================================

median_price = df["SalePrice"].median()

df["Expensive"] = (
    df["SalePrice"] > median_price
).astype(int)


X = df[features]

y = df["Expensive"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 1. LOGISTIC REGRESSION
# =========================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)

logistic_prob = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 2. KNN CLASSIFIER
# =========================================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn_model.predict(
    X_test_scaled
)

knn_prob = knn_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 3. DECISION TREE CLASSIFIER
# =========================================================

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)

tree_prob = tree_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 4. RANDOM FOREST CLASSIFIER
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# MODEL DICTIONARY
# =========================================================

models = {

    "Logistic Regression": {
        "model": logistic_model,
        "pred": logistic_pred,
        "prob": logistic_prob
    },

    "KNN Classifier": {
        "model": knn_model,
        "pred": knn_pred,
        "prob": knn_prob
    },

    "Decision Tree": {
        "model": tree_model,
        "pred": tree_pred,
        "prob": tree_prob
    },

    "Random Forest": {
        "model": rf_model,
        "pred": rf_pred,
        "prob": rf_prob
    }
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)


# =========================================================
# HOUSE INPUT
# =========================================================

st.sidebar.header("🏠 House Information")

overall_qual = st.sidebar.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.sidebar.number_input(
    "Living Area",
    min_value=300,
    max_value=5000,
    value=1500
)

garage_cars = st.sidebar.number_input(
    "Garage Cars",
    min_value=0,
    max_value=5,
    value=2
)

total_bsmt_sf = st.sidebar.number_input(
    "Basement Area",
    min_value=0,
    max_value=5000,
    value=1000
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

full_bath = st.sidebar.number_input(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

bedroom_abvgr = st.sidebar.number_input(
    "Bedrooms",
    min_value=0,
    max_value=10,
    value=3
)

lot_area = st.sidebar.number_input(
    "Lot Area",
    min_value=500,
    max_value=50000,
    value=8000
)


# =========================================================
# NEW HOUSE DATA
# =========================================================

input_data = pd.DataFrame({

    "OverallQual": [overall_qual],

    "GrLivArea": [gr_liv_area],

    "GarageCars": [garage_cars],

    "TotalBsmtSF": [total_bsmt_sf],

    "YearBuilt": [year_built],

    "FullBath": [full_bath],

    "BedroomAbvGr": [bedroom_abvgr],

    "LotArea": [lot_area]

})


# =========================================================
# PREDICTION
# =========================================================

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict"
)


# =========================================================
# SELECT MODEL
# =========================================================

selected_model = models[model_choice]

model = selected_model["model"]

pred = selected_model["pred"]

prob = selected_model["prob"]


# =========================================================
# METRICS FUNCTION
# =========================================================

def calculate_metrics(y_true, y_pred, y_prob):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )


# =========================================================
# CURRENT MODEL METRICS
# =========================================================

accuracy, precision, recall, f1, roc_auc = calculate_metrics(
    y_test,
    pred,
    prob
)


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.header(f"📊 {model_choice}")


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if model_choice in [
        "Logistic Regression",
        "KNN Classifier"
    ]:

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

    else:

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]


    st.subheader("🔮 Prediction")


    if prediction == 1:

        st.success(
            f"💰 Expensive House\n\n"
            f"Probability: {probability:.2%}"
        )

    else:

        st.info(
            f"🏠 Not Expensive House\n\n"
            f"Probability: {probability:.2%}"
        )


# =========================================================
# METRICS
# =========================================================

st.subheader("📈 Model Performance")


col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "Accuracy",
    f"{accuracy:.3f}"
)

col2.metric(
    "Precision",
    f"{precision:.3f}"
)

col3.metric(
    "Recall",
    f"{recall:.3f}"
)

col4.metric(
    "F1-Score",
    f"{f1:.3f}"
)

col5.metric(
    "ROC-AUC",
    f"{roc_auc:.3f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("🎯 Confusion Matrix")


cm = confusion_matrix(
    y_test,
    pred
)


fig, ax = plt.subplots()

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Expensive",
        "Expensive"
    ]
)

display.plot(
    ax=ax
)

ax.set_title(
    f"{model_choice} - Confusion Matrix"
)

st.pyplot(fig)


# =========================================================
# ROC CURVE
# =========================================================

st.subheader("📈 ROC Curve")


fpr, tpr, thresholds = roc_curve(
    y_test,
    prob
)


fig, ax = plt.subplots()

ax.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    f"{model_choice} - ROC Curve"
)

ax.legend()

st.pyplot(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if model_choice in [
    "Decision Tree",
    "Random Forest"
]:

    st.subheader("🌳 Feature Importance")


    importance = pd.DataFrame({

        "Feature": features,

        "Importance": model.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )


    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )


    st.dataframe(
        importance,
        use_container_width=True
    )


# =========================================================
# LOGISTIC COEFFICIENTS
# =========================================================

if model_choice == "Logistic Regression":

    st.subheader(
        "📊 Logistic Regression Coefficients"
    )


    coefficients = pd.DataFrame({

        "Feature": features,

        "Coefficient": logistic_model.coef_[0]

    }).sort_values(
        "Coefficient",
        ascending=False
    )


    st.bar_chart(
        coefficients.set_index(
            "Feature"
        )
    )


    st.dataframe(
        coefficients,
        use_container_width=True
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.divider()

st.header("🏆 Model Comparison")


comparison = []


for name, info in models.items():

    acc, prec, rec, f1_score_value, auc = calculate_metrics(
        y_test,
        info["pred"],
        info["prob"]
    )

    comparison.append({

        "Model": name,

        "Accuracy": acc,

        "Precision": prec,

        "Recall": rec,

        "F1-Score": f1_score_value,

        "ROC-AUC": auc

    })


comparison_df = pd.DataFrame(
    comparison
)


st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True
)


# =========================================================
# BEST MODEL
# =========================================================

best_model = comparison_df.loc[
    comparison_df["F1-Score"].idxmax()
]


st.success(
    f"🏆 Best Model based on F1-Score: "
    f"{best_model['Model']}"
)


# =========================================================
# DATASET
# =========================================================

st.divider()

st.subheader("📁 Dataset")

st.write(
    f"Median SalePrice used as classification threshold: "
    f"${median_price:,.0f}"
)


st.dataframe(
    df,
    use_container_width=True
)
