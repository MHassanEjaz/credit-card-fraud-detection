# Credit Card Fraud Detection

A Machine Learning project that detects fraudulent credit card transactions using classification algorithms, with a live interactive Streamlit web app for real-time and batch predictions.

## Live Demo
[Try the app here](https://your-streamlit-link.streamlit.app)

## Dataset
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- Only 0.17% are fraudulent (highly imbalanced)
- Features V1-V28 are PCA-transformed for confidentiality

## Tech Stack
- Language: Python
- ML Libraries: Scikit-learn, Imbalanced-learn (SMOTE)
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit

## Approach
1. Preprocessed and scaled Amount and Time features
2. Handled severe class imbalance using SMOTE oversampling
3. Trained and compared three classifiers: Naive Bayes, KNN, Random Forest
4. Evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC
5. Selected Random Forest as the best-performing model
6. Built a Streamlit app for real-time and batch CSV fraud prediction

## Results
| Metric | Score |
|---|---|
| Accuracy | 99.94% |
| Precision | 92.00% |
| Recall | 72.63% |
| F1 Score | 81.18% |

## How to Run Locally
git clone https://github.com/MHassanEjaz/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py

## Author
Muhammad Hassan
[LinkedIn](https://www.linkedin.com/in/muhammad-hassanofficial/) | [GitHub](https://github.com/MHassanEjaz)
