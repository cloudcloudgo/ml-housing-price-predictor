import pandas as pd
from sklearn.model_selection import train_test_split


#load data csv files
X_full = pd.read_csv('./input/train.csv', index_col='Id')
X_test_full = pd.read_csv('./input/test.csv', index_col='Id')


# Remove rows with missing target values
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
#select target column
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#select only numeric data types
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

#split training data 80% - 20% to train and validate the model
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# print(X_train.head())

# Step 1: Preliminary investigate

print(X_train.shape)
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# ## Step 1 : Evaluate several datasets and select the best one to be used for prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from functools import reduce

# model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
# model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

# models = [("model_1", model_1), ("model_2", model_2), ("model_3", model_3), ("model_4", model_4), ("model_5", model_5)]

# best model as per the previous exercise
model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)

# define a score dataset method to score dataset based on its performance with a given model
def score_dataset(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


# Step 2: Drop columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

mae = score_dataset(model, reduced_X_train, reduced_X_valid, y_train, y_valid)

print("MAE from dropping columns with missing values:" + str(mae))

# step 3: Imputation
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

mae_imputed = score_dataset(model, imputed_X_train, imputed_X_valid, y_train, y_valid)
print("MAE from imputation while training a model:" + str(mae_imputed))


# Step 4: Imputation on only Lot Frontage column
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

#drop non essential columns
cols_with_missing_garage = ["GarageYrBlt", "MasVnrArea"]
reduced_X_train_plus = X_train_plus.drop(cols_with_missing_garage, axis=1)
reduced_X_valid_plus = X_valid_plus.drop(cols_with_missing_garage, axis=1)

#impute essential columns
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(reduced_X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(reduced_X_valid_plus))

imputed_X_train_plus.columns = reduced_X_train_plus.columns
imputed_X_valid_plus.columns = reduced_X_valid_plus.columns

mae_plus = score_dataset(model, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
print("MAE from imputation while training a model:" + str(mae_plus))


#Step B test data curation

X_test =  X_test.drop(cols_with_missing_garage, axis=1)
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

imputed_X_test.columns = X_test.columns

preds_test = model.predict(imputed_X_test)

output = pd.DataFrame({'Id': imputed_X_test.index,
                       'SalePrice': preds_test})


print(output.head())