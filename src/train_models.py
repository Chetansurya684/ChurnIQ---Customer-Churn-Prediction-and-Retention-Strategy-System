from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
BASE=Path(__file__).resolve().parents[1]; DATA=BASE/'data/customer_churn.csv'; MD=BASE/'models'; MD.mkdir(exist_ok=True)
df=pd.read_csv(DATA); X=df.drop(columns=['customer_id','churn']); y=df.churn.astype(int)
num=['tenure','monthly_charges','senior_citizen']; cat=[c for c in X.columns if c not in num]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_split=4,class_weight='balanced',random_state=42,n_jobs=-1),'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=220,learning_rate=.06,max_leaf_nodes=20,random_state=42)}
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); results=[]; trained={}
for name,est in models.items():
    pipe=Pipeline([('preprocessor',pre),('model',est)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv); prob=pipe.predict_proba(Xv)[:,1]
    results.append({'model':name,'accuracy':accuracy_score(yv,pred),'precision':precision_score(yv,pred,zero_division=0),'recall':recall_score(yv,pred,zero_division=0),'f1':f1_score(yv,pred,zero_division=0),'roc_auc':roc_auc_score(yv,prob)}); trained[name]=pipe
metrics=pd.DataFrame(results).sort_values('roc_auc',ascending=False).reset_index(drop=True); best=metrics.iloc[0].model; pipe=trained[best]
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); cv_scores=cross_val_score(pipe,X,y,cv=cv,scoring='roc_auc',n_jobs=-1)
joblib.dump(pipe,MD/'best_model.joblib'); est=pipe.named_steps['model']; names=pipe.named_steps['preprocessor'].get_feature_names_out(); imp=est.feature_importances_ if hasattr(est,'feature_importances_') else np.abs(est.coef_[0]) if hasattr(est,'coef_') else np.zeros(len(names)); pd.DataFrame({'feature':names,'importance':imp}).sort_values('importance',ascending=False).to_csv(MD/'feature_importance.csv',index=False)
meta={'best_model':best,'metrics':metrics.to_dict('records'),'cv_roc_auc_mean':float(cv_scores.mean()),'cv_roc_auc_std':float(cv_scores.std()),'features':list(X.columns),'classification_report':classification_report(yv,pipe.predict(Xv),output_dict=True,zero_division=0)}; (MD/'model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print(metrics.to_string(index=False)); print(f'Best model: {best}'); print(f'5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
