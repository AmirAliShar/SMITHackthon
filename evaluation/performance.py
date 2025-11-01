import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.decomposition import PCA

sns.set(style="whitegrid")


#Classification
def evaluate_classification_model(model, X_test, y_test):
    preds = model.predict(X_test)
    results = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, average="weighted", zero_division=0))
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return results, fig


# Regression
def evaluate_regression_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2_Score": float(r2)}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, preds, alpha=0.6)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    plt.tight_layout()
    return results, fig


# Helper: PCA 2D plot
def plot_pca_2d(X, labels, title="PCA 2D Cluster Plot"):
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(Xp[mask, 0], Xp[mask, 1], label=str(lab), alpha=0.7, s=40, c=[palette[i]])
    ax.legend(title="Cluster")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    return fig


# KMeans evaluation 
def evaluate_kmeans(X, labels):
    # Silhouette (only if more than 1 cluster and not all noise)
    valid_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = None
    if valid_clusters > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    results = {
        "n_clusters_detected": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "silhouette_score": sil
    }

    fig = plot_pca_2d(X, labels, title="KMeans Clusters (PCA 2D)")
    return results, fig


# DBSCAN evaluation
def evaluate_dbscan(X, labels):
    # Count noise
    n_noise = int(sum(labels == -1))
    valid_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = None
    if valid_clusters > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    results = {
        "n_clusters_detected": int(valid_clusters),
        "noise_points": int(n_noise),
        "silhouette_score": sil
    }

    fig = plot_pca_2d(X, labels, title="DBSCAN Clusters (PCA 2D)")
    return results, fig


#Agglomerative evaluation
def evaluate_agglomerative(X, labels):
    valid_clusters = len(set(labels))
    sil = None
    if valid_clusters > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    results = {
        "n_clusters_detected": int(valid_clusters),
        "silhouette_score": sil
    }

    fig = plot_pca_2d(X, labels, title="Agglomerative Clusters (PCA 2D)")
    return results, fig
