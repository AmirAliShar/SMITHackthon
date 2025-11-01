from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_random_forest_classifier(
    X_train, y_train,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="gini"
):
    """
    Train Random Forest Classifier with key parameters.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(
    X_train, y_train,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="squared_error"
):
    """
    Train Random Forest Regressor with key parameters.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
