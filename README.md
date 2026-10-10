# 🤖 Machine Learning Model Explorer

A **Streamlit-based interactive machine learning application** that allows users to explore, train, and evaluate machine learning models using their own CSV datasets. The application simplifies the machine learning workflow by providing data preprocessing, model selection, configurable parameters, evaluation metrics, and visualizations through a user-friendly interface.

## 🚀 Features

- **CSV Dataset Upload:** Upload a dataset and preview its contents, rows, columns, and dimensions.
- **Automated Data Preprocessing:** Handle missing values, encode categorical features, and scale numerical features.
- **Supervised Learning:**
  - **Classification:** Random Forest, Decision Tree, and Support Vector Machine (SVM).
  - **Regression:** Random Forest, Decision Tree, and Support Vector Machine (SVM).
- **Unsupervised Learning:**
  - K-Means Clustering
  - DBSCAN Clustering
  - Agglomerative Clustering
- **Customizable Model Parameters:** Adjust algorithm settings through the interactive sidebar.
- **Model Evaluation:**
  - Classification: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
  - Regression: MAE, MSE, RMSE, and R² score.
  - Clustering: Number of detected clusters, silhouette score where applicable, and noise points for DBSCAN.
- **Interactive Visualizations:** View evaluation charts, confusion matrices, regression plots, and 2D PCA cluster visualizations.
- **Session State Management:** Keep preprocessed data available while working with the application.

## 🛠️ Technology Stack

- **Language:** Python
- **Web Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Environment and Dependency Management:** uv

## ⚙️ How It Works

1. Upload a CSV dataset.
2. Preview the dataset and select a learning approach.
3. For supervised learning, select the target column and choose classification or regression.
4. Configure the model parameters.
5. Preprocess the data using the application's preprocessing workflow.
6. Train the selected model or run a clustering algorithm.
7. Review the evaluation metrics and visualizations.

## 📂 Project Structure

```text
SMITHackthon/
├── cleaning/
│   └── preprocessing.py
├── evaluation/
│   └── performance.py
├── models/
│   ├── randomforest.py
│   ├── decisiontree.py
│   ├── svm.py
│   ├── kmeans.py
│   ├── dbscan.py
│   └── alg.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## 💻 Installation and Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)

### 1. Clone the repository

```bash
git clone https://github.com/AmirAliShar/SMITHackthon.git
cd SMITHackthon
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the application

```bash
uv run streamlit run main.py
```

Streamlit will provide a local URL in the terminal. Open it in your browser to use the application.

## 🎯 Project Goal

The goal of this project is to make experimenting with machine learning models more accessible by bringing dataset preprocessing, model training, evaluation, and visualization together in one interactive application.

It is useful for students, beginners, and data science learners who want to experiment with different machine learning algorithms without building a separate interface for each model.

## 🔮 Future Improvements

- Add downloadable predictions and evaluation reports.
- Support additional machine learning algorithms.
- Improve preprocessing with configurable encoding and imputation strategies.
- Add model comparison and experiment history.
- Improve dataset validation and error handling.
- Add model saving and loading functionality.

## 👨‍💻 Author

**Amir Ali**

- GitHub: [@AmirAliShar](https://github.com/AmirAliShar)
- LinkedIn: [Amir Ali](https://www.linkedin.com/in/amir-ali-945a53242/)

---

*An interactive machine learning project for exploring supervised learning, unsupervised learning, and model evaluation.*


<img width="1912" height="895" alt="image" src="https://github.com/user-attachments/assets/4d6c9ba7-2171-42a2-95a0-32983428bcc4" />
<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/38ebb685-c438-4402-af86-7dae8716dfd0" />
<img width="1491" height="326" alt="image" src="https://github.com/user-attachments/assets/25e3a0b9-b62f-47b4-8ae7-93a6073b1cae" />
v<img width="1902" height="895" alt="image" src="https://github.com/user-attachments/assets/5c871ea6-92ad-494a-b7c3-5ee0b26c5b51" />
