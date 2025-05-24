#load all required libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import pickle

#load data
df = pd.read_csv('Exercise.csv')

#select needed columns
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp']
cat_cols = ['Gender']

#split into target and features first
X = df.drop(columns=['User_ID','Calories'])
y = np.sqrt(df['Calories'])

#import the required library for splitting into train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build Preprocessor Pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

PCA_transformer = Pipeline(steps=[
    ('pca', PCA())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
    ('pca', PCA_transformer, ['Duration', 'Heart_Rate', 'Body_Temp'])
])

#Build model
model = XGBRegressor(random_state=42)

#Build pipline
pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('Regressor', model)
    ])
pipeline.fit(X_train, y_train)

#Save the model
with open('Regressor.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print('Model saved succesfully')

#sample data
sample = pd.DataFrame([{
    'Age': 41,
    'Height': 175,
    'Weight': 85,
    'Duration': 25,
    'Heart_Rate': 10,
    'Body_Temp': 40.7,
    'Gender': 'male'
}])

#predict sample data
sqrt_pred = pipeline.predict(sample)[0]

#convert to actual calories after sqrt trandormation
pred_calories = np.round(sqrt_pred**2, 2)
print(f'Predicted Calories Burnt is {pred_calories}')
