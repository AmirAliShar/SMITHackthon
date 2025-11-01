from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters=3, init="k-means++", n_init=10, random_state=42):
    km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km
