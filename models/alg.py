from sklearn.cluster import AgglomerativeClustering

def run_agglomerative(X, n_clusters=3, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model
