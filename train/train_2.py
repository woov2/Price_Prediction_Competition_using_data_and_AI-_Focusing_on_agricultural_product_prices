import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

################################## 무 #######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_무.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]
state = 42

params = {'max_depth': 3,
 'learning_rate': 0.01842776597084574,
 'n_estimators': 196,
 'colsample_bytree': 0.502795884302695,
 'subsample': 0.6479858916463348,
 'min_child_weight': 1,
 'reg_alpha': 0.03334011653368006,
 'reg_lambda': 0.09097354296782251,
 'seed': 42}

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
model = XGBRegressor(**params)
model.fit(data, value)

os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

with open('models/xgb_model_무.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scalers/scaler_무.pkl', 'wb') as f:
    pickle.dump(scaler, f)

################################ 대파 #######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_대파.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]

params = {'max_depth': 11,
 'learning_rate': 0.023881266701407652,
 'n_estimators': 874,
 'colsample_bytree': 0.9036676990841959,
 'subsample': 0.6464147332115494,
 'min_child_weight': 2,
 'reg_alpha': 0.0018537127129309207,
 'reg_lambda': 0.013667102910223567,
 'seed': 42}

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
model = XGBRegressor(**params)
model.fit(data, value)

with open('models/xgb_model_대파.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scalers/scaler_대파.pkl', 'wb') as f:
    pickle.dump(scaler, f)

################################ 감자 #######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_감자.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]

params = {'max_depth': 4,
 'learning_rate': 0.012982573713268443,
 'n_estimators': 333,
 'colsample_bytree': 0.5619466288097043,
 'subsample': 0.5773121540947779,
 'min_child_weight': 1,
 'reg_alpha': 0.0036804967411978597,
 'reg_lambda': 0.031642831595928005,
 'seed': 42}

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
model1 = XGBRegressor(**params)
model1.fit(data, value)

model2 = ExtraTreesRegressor(random_state=state)
model2.fit(data, value)

with open('models/xgb_model_감자.pkl', 'wb') as f:
    pickle.dump(model1, f)

with open('models/et_model_감자.pkl', 'wb') as f:
    pickle.dump(model2, f)

with open('scalers/scaler_감자.pkl', 'wb') as f:
    pickle.dump(scaler, f)

################################ 상추 #######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_상추.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]

params = {'max_depth': 34,
 'learning_rate': 0.23119043973000283,
 'n_estimators': 746,
 'colsample_bytree': 0.6434650083980358,
 'subsample': 0.9591076992587381,
 'min_child_weight': 3,
 'reg_alpha': 0.04513024530971878,
 'reg_lambda': 0.007868385954555494,
 'seed': 42}

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
model1 = XGBRegressor(**params)
model1.fit(data, value)

model2 = ExtraTreesRegressor(random_state=state)
model2.fit(data, value)

with open('models/xgb_model_상추.pkl', 'wb') as f:
    pickle.dump(model1, f)

with open('models/et_model_상추.pkl', 'wb') as f:
    pickle.dump(model2, f)

with open('scalers/scaler_상추.pkl', 'wb') as f:
    pickle.dump(scaler, f)

################################ 배 #######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_배.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
state = 42

model = RANSACRegressor(random_state=state)
model.fit(data, value)

with open('models/ransac_model_배.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scalers/scaler_배.pkl', 'wb') as f:
    pickle.dump(scaler, f)

############################### 건고추 ######################################

preprocessed_data = pd.read_csv('./preprocessing_train/preprocessing_건고추.csv')
X, y = preprocessed_data.iloc[:,:-3], preprocessed_data.iloc[:,-3:]

data = X
value = y
scaler = StandardScaler()
value = scaler.fit_transform(value)
model = RANSACRegressor(random_state=state)
model.fit(data, value)

with open('models/ransac_model_건고추.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scalers/scaler_건고추.pkl', 'wb') as f:
    pickle.dump(scaler, f)
