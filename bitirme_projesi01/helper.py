import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

def plot_outliers(dataframe, col_name):
    sns.boxplot(x=dataframe[col_name])
    plt.title(f"Outlier Check: {col_name}")
    plt.show()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

def scale_features(dataframe, num_cols, method="standard"):
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def data_prep(dataframe, scale_method="standard", q1=0.10, q3=0.90, drop_first=False, verbose=True):
    # Kolon türlerini al
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
      
    # Aykırı değer kontrolü ve eşiklerle değiştirme
    for col in num_cols:
        if verbose:
            print(f"{col} için aykırı değer kontrolü yapılıyor...")
            plot_outliers(dataframe, col)

        if check_outlier(dataframe, col, q1=q1, q3=q3):
            if verbose:
                print(f"Aykırı değer bulundu! {col} için eşik değerlerle değiştiriliyor...")
            replace_with_thresholds(dataframe, col)
        else:
            if verbose:
                print(f"{col} için aykırı değer bulunamadı.")

        if verbose:
            plot_outliers(dataframe, col)

    # Sayısal değişkenleri ölçekle
    dataframe = scale_features(dataframe, num_cols, method=scale_method)

    # One-hot encoding
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=drop_first)

    return dataframe, cat_cols, num_cols