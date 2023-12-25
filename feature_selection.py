import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, mutual_info_regression


if __name__ == "__main__":

#     train_df = pd.read_csv("train.csv")
#     preprocessed_train_df = train_df.copy()
#     preprocessed_train_df['missing_Product_Category_2'] = preprocessed_train_df['Product_Category_2'].isnull().astype(int)
#     preprocessed_train_df['missing_Product_Category_3'] = preprocessed_train_df['Product_Category_3'].isnull().astype(int)

#     # # Imput the missing value with constant 0
#     # preprocessed_train_df['Product_Category_2'].fillna(0, inplace=True) 
#     # preprocessed_train_df['Product_Category_3'].fillna(0, inplace=True)
#     # # And also apply to test_df
#     # preprocessed_test_df['Product_Category_2'].fillna(0, inplace=True) 
#     # preprocessed_test_df['Product_Category_3'].fillna(0, inplace=True)

#     # Gender: Binary encode 'Gender'
#     preprocessed_train_df['Gender']=preprocessed_train_df['Gender'].map({'M': 0, 'F': 1})

#     # Age: Ordinal encode 'Age'
#     age_mapping = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
#     preprocessed_train_df['Age'] = preprocessed_train_df['Age'].map(age_mapping)

#     # City_Category: One-hot encode 'City_Category'
#     train_city_category_dummies = pd.get_dummies(preprocessed_train_df['City_Category'], prefix='City').astype(int)
#     preprocessed_train_df = pd.concat([preprocessed_train_df, train_city_category_dummies], axis=1)

#     # Stay_In_Current_City_Years: Replace '4+' with '4' and convert to integer
#     preprocessed_train_df['Stay_In_Current_City_Years'] = preprocessed_train_df['Stay_In_Current_City_Years'].str.replace('+','').astype(int)

#     # Drop the original categorical columns after encoding
#     preprocessed_train_df.drop(['City_Category'], axis=1, inplace=True)
    preprocessed_train_df = pd.read_csv("train.csv")
    preprocessed_train_df = preprocessed_train_df.drop(columns=["missing_Product_Category_2", "missing_Product_Category_3"])
    X = preprocessed_train_df.drop(columns=['Purchase', 'Product_ID'], axis=1)
    y = list(preprocessed_train_df['Purchase'])
    # k = 4 tells four top features to be selected
    # Score function Chi2 tells the feature to be selected using Chi Square
    test = SelectKBest(score_func=mutual_info_regression,k=8)
    fit = test.fit(X, y)
    X_new=test.fit_transform(X, y)
    X_new.to_csv("features.csv")
