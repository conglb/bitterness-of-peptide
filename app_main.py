from os import listdir
from os.path import join
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.express as px
import streamlit as st
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO


@dataclass
class UserInput:
    SRC_FILE_PATH: str  # source csv file
    GROUP: str  # heuristic group of attributes
    STATE: str  # ["At Anchor", "In Port", "Drifting", "At Sea", "Manoeuvring"]
    DATA_TYPE: str
    AVAILABILITY_LOW: float
    AVAILABILITY_HIGH: float
    UNI_OD_METHOD: str  # Univariate outlier detection method
    IMPUTATION_METHOD: str


def preprocess(df, preserve_columns: list[str], standardize=False) -> pd.DataFrame:
    """
    1. set index to be time
    2. select by name of columns
    3. remove outliers
    4. time linear imputation
    5. Drop row containing <NA>
    """
    df = df.copy()
    # Middle & High Frquency Data
    if "entry_date" in df: timestamp = pd.to_datetime(df["entry_date"])
    elif "Timestamp (UTC)" in df: timestamp = pd.to_datetime(df["Timestamp (UTC)"])
    else: raise "No timestamp column found"
    df = df.set_index(timestamp)
    if "sog_kn**2" in preserve_columns:
        df["sog_kn**2"] = df["sog_kn"] ** 2
    if "sog_kn**0.5" in preserve_columns:
        df["sog_kn**0.5"] = df["sog_kn"] ** 0.5
    df = df[preserve_columns]
    df = univariate_outlier_detection(df, method="Inner Fence")
    df = impute(df, method="Time Linear")
    df = df.dropna()
    df = df.reindex(sorted(df.columns), axis=1)
    if standardize: df[:] = MinMaxScaler().fit_transform(df) 
    return df


def elbow(df):
    """Elbow algorithm :ref(https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)

    Args:
        df (Dataframe): Dataset after preprocessed

    Returns:
        Figure: figure shows error according to k
    """
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 12))

    visualizer.fit(df.to_numpy())   # Fit the data to the visualizer
    visualizer.finalize()
    # visualizer.show()   # Finalize and render the figure
    return plt.gcf()


def mean_absolute_error(groundtruth: pd.Series, prediction: np.array) -> pd.Series:
    """Calculate MSE from ground truth and prediction.

    Args:
        groundtruth (pd.Series): The truth
        prediction (np.array): The prediction

    Returns:
        pd.Series: a number indicates the error
    """
    groundtruth = groundtruth.copy()
    error = (groundtruth - prediction).abs().mean()
    return error


def get_model(df: pd.DataFrame, target_column: str, prediction_model: str):
    """Train the model and return its performance

    Args:
        df (pd.DataFrame): Dataset after preprocessed
        target_column (str): The feature need to predict
        prediction_model (str): Name of model

    Returns:
        tuple (error, score): MAE and R^2 Score shows model's performance
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # change number here
    X_train, X_test = X_train.sort_index(), X_test.sort_index()
    y_train, y_test = y_train.sort_index(), y_test.sort_index()

    if prediction_model == "Linear Regression":
        model = LinearRegression()
    elif prediction_model == "SVM":
        model = svm.SVR()
    elif prediction_model == "MLP":
        model = MLPRegressor(max_iter=2000, hidden_layer_sizes=10, alpha=0.1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_absolute_error(y_test, prediction)

    # Plot target_column
    df_plot = pd.DataFrame(columns=[target_column, "prediction"])
    df_plot[target_column] = y_test
    df_plot["prediction"] = prediction
    # fig = px.line(df_plot, markers=True)
    fig = px.line(df_plot)
    fig.update_layout(title_text=target_column, title_x=0.5)
    st.write(fig)

    if prediction_model == 'Linear Regression':
        coefs = pd.DataFrame(model.coef_, columns=['importance'], index=X_train.columns)
        fig = px.bar(coefs, x='importance', title='Feature importance', orientation='h', range_x=[-1,1])
        fig.update_layout(title_x=0.5)
        st.write(fig)

        # df_effect = X_train.mul(model.coef_)
        # fig = px.box(df_effect, title='Feature effect')
        # fig.update_layout(title_x=0.5)
        # st.write(fig)
    
    score = model.score(X_test, y_test)
    return error, score

def get_embedding():
    return 0


def get_sidebar():
    return 0

def main():
    # Build side bar for selection
    # Select file
    directory = join("data", "b1")
    files = listdir(join(directory))
    file = st.sidebar.selectbox("Select a file from middle frequency", files, index=1, label_visibility='visible')

    # Select target to the prediction model
    target_column = st.sidebar.radio(
        "The target of the prediction model", ['bitterness'], index=0)

    # Select embedding model (prediction model)
    embedding_model = st.sidebar.radio("Select a embedding model", [
                                        "BiLSTM", "Bert", "ProteinBert", "Pep2Vec"])

    # Select regression model (prediction model)
    prediction_model = st.sidebar.radio("Select a classification model", [
                                        "Linear Regression", "SVM", "MLP"])

    # Whether apply cluster
    st.sidebar.markdown("---")
    is_standardize = st.sidebar.checkbox("Standardize data")
    is_cluster = st.sidebar.checkbox("Apply cluster", value=False)

    # Whether visualize elbow method for PCA clustering
    is_elbow = st.sidebar.checkbox(
        "Show elbow method for PCA clustering", value=False)

    # Streamlit Main Page
    st.markdown("# Prediction of Peptides")

    # Load file and preprocess file
    records = list(SeqIO.parse(join(directory, file), "fasta"))
    data = [[str(record.seq), int(record.id == 'Positive')] for record in records]
    df = pd.DataFrame(data, columns=['seq', 'bitter'])
    #st.markdown("## Data before Preprocessed")
    #st.write(df)
    #with st.expander("Data before Preprocessed"):
    st.write(df)
    st.write(f"Shape of the table: {df.shape}")
    #df_all = preprocess(df, input_columns + [target_column], standardize=is_standardize)
    #df = df_all.drop(target_column, axis=1)
    #with st.expander("## Data after Preprocessed"):
    #    st.write(df_all)
    #    st.write(f"Shape of the table: {df.shape}")

    if not is_cluster:
        # Apply prediction model on all data points
        error, score = get_model(df, target_column, prediction_model)
        col1, col2 = st.columns(2)
        col1.metric(
            "Mean Squared Error",
            f"{error:.5f}",
            f"{0}",
        )
        col2.metric(
            "Score",
            f"{score*100:.3f}%",
            f"{0}",
        )

    if is_cluster:
        # 3 cluster by elbow method
        st.markdown("## KNN Cluster (always set to 3 clusters)")
        X = df.to_numpy()
        kmeans = KMeans(n_clusters=3).fit(X)
        labels = kmeans.labels_

        st.markdown("### Centers of each Cluster")
        st.write(pd.DataFrame(columns=df.columns, data=kmeans.cluster_centers_))

        cluster1 = df[labels == 0]
        cluster2 = df[labels == 1]
        cluster3 = df[labels == 2]

        st.markdown("## Plot 3 clusters (after PCA)")
        pca = PCA(n_components=2)
        pca.fit(X)
        pca_cluster1 = pca.transform(cluster1.to_numpy())
        pca_cluster2 = pca.transform(cluster2.to_numpy())
        pca_cluster3 = pca.transform(cluster3.to_numpy())

        df1_tmp = pd.DataFrame(data=pca_cluster1, columns=["x", "y"])
        df1_tmp["cluster"] = "1"
        df2_tmp = pd.DataFrame(data=pca_cluster2, columns=["x", "y"])
        df2_tmp["cluster"] = "2"
        df3_tmp = pd.DataFrame(data=pca_cluster3, columns=["x", "y"])
        df3_tmp["cluster"] = "3"
        df_pxplot = pd.concat([df1_tmp, df2_tmp, df3_tmp],
                              axis=0, ignore_index=True)
        fig = px.scatter(df_pxplot, x="x", y="y", color="cluster")
        st.write(fig)

        st.markdown("## Regression Predction")
        cluster1 = df_all[labels == 0]
        cluster2 = df_all[labels == 1]
        cluster3 = df_all[labels == 2]

        st.markdown("### Cluster 1")
        error1, score1 = get_model(cluster1, target_column, prediction_model)
        st.write(f"Error of {target_column}: {error1:.3f}")
        st.write("Score of the prediction: %.5f" % score1)

        st.markdown("### Cluster 2")
        error2, score2 = get_model(cluster2, target_column, prediction_model)
        st.write(f"Error of {target_column}: {error2:.3f}")
        st.write("Score of the prediction: %.5f" % score2)

        st.markdown("### Cluster 3")
        error3, score3 = get_model(cluster3, target_column, prediction_model)
        st.write(f"Error of {target_column}: {error3:.3f}")
        st.write("Score of the prediction: %.5f" % score3)

    if is_elbow:
        st.markdown("## Elbow Method")
        with st.empty():
            st.markdown("> Take some time to generate plot...")
            elbow_fig = elbow(df)
            st.write(elbow_fig)


if __name__ == "__main__":
    st.set_page_config(
        "Bitter Peptides classification",
        "📊",
        initial_sidebar_state="expanded",
        layout="centered",
    ) 
    user_input = get_sidebar()
    main()
