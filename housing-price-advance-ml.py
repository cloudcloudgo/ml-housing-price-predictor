import pandas as pd
from sklearn.model_selection import train_test_split


#load csv files
X_full = pd.read_csv('./input/train.csv', index_col='Id')
X_test_full = pd.read_csv('./input/test.csv', index_col='Id')

#select target column
y = X_full.SalePrice

#define features of the data
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#select only the required columns from the data
X = X_full[features].copy()
X_test = X_test_full[features].copy()

#split training data 80% - 20% to train and validate the model
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print(X_train.head())

## Step 1 : Evaluate several models and select the best one to be used for prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from functools import reduce

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [("model_1", model_1), ("model_2", model_2), ("model_3", model_3), ("model_4", model_4), ("model_5", model_5)]


# define a score model method to evaluate the model performance
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

# run a for loop through all models
for i in range(0, len(models)):
    model_name, model = models[i]   
    mae = score_model(model)
    print("Model %d MAE: %d" % (i+1, mae))

mae_list = [(model_name, score_model(model, X_train, X_valid, y_train, y_valid)) for model_name, model in models]

def min_finder(a,b):
    (model_a, mae_a) = a
    (model_b, mae_b) = b
    if(mae_a < mae_b):
        return a
    else:
        return b
    
(model, min_mae) = reduce(min_finder, mae_list)


print("Best model is %s with MAE %d" % (model, min_mae))