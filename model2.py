import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

forest_fires = fetch_ucirepo(id=162) 
  

X = forest_fires.data.features 
y = forest_fires.data.targets 
  
# metadata 
# print(forest_fires.metadata) 
  
# variable information 
# print(forest_fires.variables) 

y.name = 'BurnedArea'


df = pd.concat([X, y], axis=1)

print(df.head())

categorical_features = ['month', 'day']
one_hot_encoder = OneHotEncoder()

df['area'] = np.log(df['area'] + 1)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)
    ],
    remainder='passthrough'  
)

X = df.drop('area', axis=1)
y = df['area']
X_preprocessed = preprocessor.fit_transform(X)
feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) + df.columns.drop(categorical_features + ['area']).tolist()

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

xgb_pipeline = make_pipeline(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))

xgb_pipeline.fit(X_train, y_train)

y_pred = xgb_pipeline.predict(X_test)

y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
print(f'RMSE: {rmse}')

xgb_regressor = xgb_pipeline.named_steps['xgbregressor']


feature_importances = xgb_regressor.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print(importance_df)


model_filename = 'xgb_regressor_model.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(xgb_pipeline, file)
