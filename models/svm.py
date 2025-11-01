from sklearn.svm import SVC, SVR

def train_svm_classifier(
    X_train, y_train,
    kernel="rbf",
    C=1.0,
    gamma="scale",
    degree=3
):
    """
    Train SVM classifier. Note: ensure data is scaled (preprocessing includes scaling).
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_svm_regressor(
    X_train, y_train,
    kernel="rbf",
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    degree=3
):
    """
    Train SVR regressor. Note: ensure data is scaled.
    """
    model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        degree=degree
    )
    model.fit(X_train, y_train)
    return model
