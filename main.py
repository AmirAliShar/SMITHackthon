import streamlit as st
import pandas as pd

from cleaning.preprocessing import preprocess_data, validate_dataframe_numeric_only
from models.randomforest import (
    train_random_forest_classifier,
    train_random_forest_regressor
)
from models.decisiontree import (
    train_decision_tree_classifier,
    train_decision_tree_regressor
)
from models.svm import (
    train_svm_classifier,
    train_svm_regressor
)
from models.kmeans import run_kmeans
from models.dbscan import run_dbscan
from models.alg import run_agglomerative

from evaluation.performance import (
    evaluate_classification_model,
    evaluate_regression_model,
    evaluate_kmeans,
    evaluate_dbscan,
    evaluate_agglomerative
)


def main():
    st.set_page_config(page_title="ML Model Explorer", layout="wide", page_icon="🤖")
    st.title("🤖 Machine Learning Model Explorer")

    # Sidebar controls
    st.sidebar.header("Controls")

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.sidebar.info("👆Upload a CSV file to begin.")
        st.stop()

    # Read dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape of dataset")
    st.write(df.shape)

    # Learning type selection
    learning_type = st.sidebar.radio("Select Learning Type", ["Supervised", "Unsupervised"])

    # Initialize session state for preprocessing results
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'preprocessing_params' not in st.session_state:
        st.session_state.preprocessing_params = {}

    if learning_type == "Supervised":
        st.sidebar.markdown("### Supervised Settings")
        target_col = st.sidebar.selectbox("Select Target Column", ["Select"] + list(df.columns))
        if target_col == "Select":
            st.sidebar.info("Select a target column to proceed")
            st.stop()

        # Show task
        task = st.sidebar.selectbox("Task", ["Classification", "Regression"])

        # Model selector
        if task == "Classification":
            model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Decision Tree", "SVM"])
        else:
            model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Decision Tree", "SVM"])

        st.sidebar.markdown("### Model Parameters")

        # Parameter
        if model_choice == "Random Forest":
            rf_n = st.sidebar.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10)
            rf_max_depth = st.sidebar.number_input("max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1)
            rf_min_split = st.sidebar.number_input("min_samples_split", min_value=2, max_value=50, value=2, step=1)
            rf_criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"] if task == "Classification" else ["squared_error", "absolute_error"])
        elif model_choice == "Decision Tree":
            dt_criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"] if task == "Classification" else ["squared_error", "friedman_mse"])
            dt_max_depth = st.sidebar.number_input("max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1)
            dt_min_split = st.sidebar.number_input("min_samples_split", min_value=2, max_value=50, value=2, step=1)
            dt_min_leaf = st.sidebar.number_input("min_samples_leaf", min_value=1, max_value=50, value=1, step=1)
        else:  # SVM
            svm_kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
            svm_C = st.sidebar.number_input("C", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, format="%.2f")
            svm_gamma = st.sidebar.selectbox("gamma", ["scale", "auto"])
            svm_degree = st.sidebar.number_input("degree (for poly)", min_value=1, max_value=10, value=3, step=1)

        # Check if preprocessing is needed
        current_params = {
            'file_hash': uploaded_file.getvalue().__hash__(),
            'learning_type': learning_type,
            'target_col': target_col,
            'task': task
        }
        
        needs_preprocessing = (
            st.session_state.preprocessed_data is None or
            st.session_state.preprocessing_params != current_params
        )

        # Preprocess button - only show if preprocessing is needed
        if needs_preprocessing:
            if st.sidebar.button("Preprocess Data"):
                st.info("Preprocessing dataset (AUTO) — encoding, scaling, duplicates, missing values...")
                try:
                    
                    X_train, X_test, y_train, y_test = preprocess_data(
                        df, task_type="supervised", target_column=target_col
                    )
                    
                    # Store in session state
                    st.session_state.preprocessed_data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                    st.session_state.preprocessing_params = current_params
                    
                    st.success("✅ Data preprocessing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()
        else:
            st.sidebar.success("✅ Data already preprocessed")

        # Train button - only enabled if preprocessing is complete
        train_disabled = st.session_state.preprocessed_data is None
        
        if st.sidebar.button("Train Model", disabled=train_disabled):
            if train_disabled:
                st.warning("Please preprocess the data first")
                st.stop()
                
            # Get preprocessed data from session state
            preprocessed = st.session_state.preprocessed_data
            X_train = preprocessed['X_train']
            X_test = preprocessed['X_test']
            y_train = preprocessed['y_train']
            y_test = preprocessed['y_test']

            # Validate numeric now (after encoding/scaling)
            is_numeric_ok, non_numeric_cols = validate_dataframe_numeric_only(X_train)
            if not is_numeric_ok:
                st.error("Training requires numeric features. Non-numeric columns found after preprocessing: " + ", ".join(non_numeric_cols))
                st.stop()

            # Train selected model
            with st.spinner("Training model..."):
                if task == "Classification":
                    if model_choice == "Random Forest":
                        maxd = None if rf_max_depth == 0 else int(rf_max_depth)
                        model = train_random_forest_classifier(
                            X_train, y_train,
                            n_estimators=int(rf_n),
                            max_depth=maxd,
                            min_samples_split=int(rf_min_split),
                            min_samples_leaf=1,
                            criterion=rf_criterion
                        )
                    elif model_choice == "Decision Tree":
                        maxd = None if dt_max_depth == 0 else int(dt_max_depth)
                        model = train_decision_tree_classifier(
                            X_train, y_train,
                            criterion=dt_criterion,
                            max_depth=maxd,
                            min_samples_split=int(dt_min_split),
                            min_samples_leaf=int(dt_min_leaf)
                        )
                    else:  # SVM
                        model = train_svm_classifier(
                            X_train, y_train,
                            kernel=svm_kernel,
                            C=float(svm_C),
                            gamma=svm_gamma,
                            degree=int(svm_degree)
                        )

                    results, fig = evaluate_classification_model(model, X_test, y_test)

                else:  # Regression
                    if model_choice == "Random Forest":
                        maxd = None if rf_max_depth == 0 else int(rf_max_depth)
                        model = train_random_forest_regressor(
                            X_train, y_train,
                            n_estimators=int(rf_n),
                            max_depth=maxd,
                            min_samples_split=int(rf_min_split),
                            min_samples_leaf=1,
                            criterion=rf_criterion
                        )
                    elif model_choice == "Decision Tree":
                        maxd = None if dt_max_depth == 0 else int(dt_max_depth)
                        model = train_decision_tree_regressor(
                            X_train, y_train,
                            criterion=dt_criterion,
                            max_depth=maxd,
                            min_samples_split=int(dt_min_split),
                            min_samples_leaf=int(dt_min_leaf)
                        )
                    else:
                        model = train_svm_regressor(
                            X_train, y_train,
                            kernel=svm_kernel,
                            C=float(svm_C),
                            epsilon=0.1,
                            gamma=svm_gamma,
                            degree=int(svm_degree)
                        )

                    results, fig = evaluate_regression_model(model, X_test, y_test)

            # Simple vertical display
            st.subheader("Evaluation Results")
            st.json(results)

            st.subheader("Visualization")
            st.pyplot(fig)

    else:
        # UNSUPERVISED
        st.sidebar.markdown("### Unsupervised Settings")
        model_choice = st.sidebar.selectbox("Choose Unsupervised Model", ["KMeans", "DBSCAN", "Agglomerative"])
        st.sidebar.markdown("### Model Parameters (only visible for chosen model)")

        if model_choice == "KMeans":
            k = st.sidebar.slider("n_clusters (k)", min_value=2, max_value=15, value=3)
            init = st.sidebar.selectbox("init", ["k-means++", "random"])
            n_init = st.sidebar.number_input("n_init", min_value=1, max_value=50, value=10)
        elif model_choice == "DBSCAN":
            eps = st.sidebar.number_input("eps", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f")
            min_samples = st.sidebar.number_input("min_samples", min_value=1, max_value=50, value=5, step=1)
        else:  # Agglomerative
            n_clusters = st.sidebar.slider("n_clusters", min_value=2, max_value=15, value=3)
            linkage = st.sidebar.selectbox("linkage", ["ward", "complete", "average", "single"])

        # Check if preprocessing is needed for unsupervised
        current_params = {
            'file_hash': uploaded_file.getvalue().__hash__(),
            'learning_type': learning_type,
            'model_choice': model_choice
        }
        
        needs_preprocessing = (
            st.session_state.preprocessed_data is None or
            st.session_state.preprocessing_params != current_params
        )

        # Preprocess button for unsupervised
        if needs_preprocessing:
            if st.sidebar.button("Preprocess Data"):
                st.info("Preprocessing dataset (AUTO) for clustering (Label Encoding + scaling)...")
                try:
                    
                    X_proc = preprocess_data(df, task_type="unsupervised", target_column=None)
                    
                    # Store in session state
                    st.session_state.preprocessed_data = {'X_proc': X_proc}
                    st.session_state.preprocessing_params = current_params
                    
                    st.success("✅ Data preprocessing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()
        else:
            st.sidebar.success("✅ Data already preprocessed")

        # clustering
        run_disabled = st.session_state.preprocessed_data is None
        
        if st.sidebar.button("Run Clustering", disabled=run_disabled):
            if run_disabled:
                st.warning("Please preprocess the data first")
                st.stop()
                
            # Get preprocessed data from session state
            preprocessed = st.session_state.preprocessed_data
            X_proc = preprocessed['X_proc']

            # Validate numeric after preprocessing
            is_numeric_ok, non_numeric_cols = validate_dataframe_numeric_only(X_proc)
            if not is_numeric_ok:
                st.error("Clustering requires numeric features. Non-numeric columns found after preprocessing: " + ", ".join(non_numeric_cols))
                st.stop()

            # Run clustering
            with st.spinner("Running clustering algorithm..."):
                if model_choice == "KMeans":
                    labels, model = run_kmeans(X_proc, n_clusters=int(k), init=init, n_init=int(n_init))
                    results, fig = evaluate_kmeans(X_proc, labels)
                elif model_choice == "DBSCAN":
                    labels, model = run_dbscan(X_proc, eps=float(eps), min_samples=int(min_samples))
                    results, fig = evaluate_dbscan(X_proc, labels)
                else:
                    labels, model = run_agglomerative(X_proc, n_clusters=int(n_clusters), linkage=linkage)
                    results, fig = evaluate_agglomerative(X_proc, labels)

            
            st.subheader("📊 Clustering Results")
            st.json(results)

            st.markdown("") 

            st.subheader("📈 Cluster Plot (PCA 2D)")
            st.pyplot(fig)
                

    # Clear preprocessing button
    if st.sidebar.button("Clear Preprocessing"):
        st.session_state.preprocessed_data = None
        st.session_state.preprocessing_params = {}
        st.success("Preprocessing cleared! Click 'Preprocess Data' to preprocess again.")
     # Add some helpful tips
    with st.expander("💡 Tips & Information"):
        st.markdown("""
        - **Preprocessing**: Handles missing values, encoding categorical variables, and feature scaling automatically
        - **Model Selection**: Choose based on your data size and problem type
        - **Parameters**: Start with default values and adjust based on results
        - **Visualization**: Results include performance metrics and visualizations
        """)


if __name__ == "__main__":
    main()