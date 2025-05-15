import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(layout='wide')
st.title("ML Notebook Explorer")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/real4_labelled.csv")
    df.columns = df.columns.str.strip()  # Strip extra spaces from column names
    return df

df = load_data()

tab1, tab2, tab3 = st.tabs(["🔵 KMeans Clustering", "🟢 XGBoost Prediction", "🔴 PCA Visualization"])

with tab1:
    st.header("KMeans Clustering")
    X = df[["Ultrasonik", "Infrared", "Magnetik", "Cahaya"]]
    k = st.slider("Number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    df["Cluster"] = clusters
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Ultrasonik", y="Infrared", hue="Cluster", palette="tab10", ax=ax1)
    st.pyplot(fig1)

with tab2:
    st.header("XGBoost Prediction")
    X = df[["Ultrasonik", "Infrared", "Magnetik", "Cahaya"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2%}")
    st.write("Sample Predictions:")
    st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head())

with tab3:
    st.header("PCA Visualization")
    features = ["Ultrasonik", "Infrared", "Magnetik", "Cahaya"]
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X.shape[1])])
    pca_df["Label"] = df["Label"]
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Label", palette="deep", ax=ax2)
    st.pyplot(fig2)
