# Customer Churn Prediction and Retention Strategy System — Advanced

Aligned with the resume project: **Python, Machine Learning, MySQL, interactive dashboard**.

## Advanced modules
1. Data preprocessing and validation
2. Logistic Regression + Random Forest + HistGradientBoosting model comparison
3. Accuracy, Precision, Recall, F1 and ROC-AUC
4. 5-fold cross-validation
5. Automatic best-model selection
6. Customer probability and Low/Medium/High risk scoring
7. Retention recommendation engine
8. Customer Risk Center with filters/search
9. Feature importance
10. MySQL persistence for customers, predictions and retention actions
11. Downloadable risk/retention report
12. Streamlit interactive dashboard

## Run
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train_models.py
python -m streamlit run app.py
```

## MySQL
Create `churn_db` in XAMPP/MySQL, then set MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE if your values differ. Run `python src\init_db.py`.

**Note:** The included data is synthetic demonstration data. Be transparent about this in academic monitoring.
