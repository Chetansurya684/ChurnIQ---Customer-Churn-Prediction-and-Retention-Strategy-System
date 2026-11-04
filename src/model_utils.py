from pathlib import Path
import json, joblib, pandas as pd
BASE_DIR=Path(__file__).resolve().parents[1]
DATA_PATH=BASE_DIR/'data/customer_churn.csv'; MODEL_DIR=BASE_DIR/'models'; MODEL_PATH=MODEL_DIR/'best_model.joblib'; META_PATH=MODEL_DIR/'model_metadata.json'; FI_PATH=MODEL_DIR/'feature_importance.csv'
FEATURES=['tenure','monthly_charges','contract','internet_service','tech_support','payment_method','senior_citizen','partner','paperless_billing']
def load_data(path=DATA_PATH): return pd.read_csv(path)
def load_artifacts():
    if not MODEL_PATH.exists(): raise FileNotFoundError('Model not trained. Run: python src\train_models.py')
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text(encoding='utf8'))
def risk_level(p): return 'HIGH' if p>=.60 else ('MEDIUM' if p>=.30 else 'LOW')
def retention_strategy(row,p):
    risk=risk_level(p); a=[]
    a += ['Contact the customer proactively within 24–48 hours.','Offer a personalized retention benefit or loyalty incentive.'] if risk=='HIGH' else (['Schedule targeted engagement or service follow-up.','Send a personalized offer based on customer usage.'] if risk=='MEDIUM' else ['Maintain service quality and regular engagement.'])
    if row.get('contract')=='Month-to-month': a.append('Consider a suitable long-term contract benefit.')
    if row.get('tech_support')=='No': a.append('Offer a technical support check-up or assistance.')
    if row.get('internet_service')=='Fiber optic': a.append('Review service quality and pricing perception.')
    if float(row.get('monthly_charges',0))>=90: a.append('Review pricing/value perception and eligible plan options.')
    return a
