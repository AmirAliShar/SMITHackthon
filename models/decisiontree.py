from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def train_decision_tree_classifier(
    X_train, y_train,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
):
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree_regressor(
    X_train, y_train,
    criterion="squared_error",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
):
    model = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
