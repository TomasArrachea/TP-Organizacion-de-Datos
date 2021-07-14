import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

## Funciones utilizadas para el preprocesamiento de cada modelo

def dummy_variables(df):
    columnas = [
        'estado_marital',
        'genero',
        'religion',
        'barrio',
        'categoria_de_trabajo',
        'rol_familiar_registrado',
        'trabajo'
    ]
    df_dummy = pd.get_dummies(df, columns=columnas, dummy_na=False, drop_first=True)
    return df_dummy

def remove_irrelevant_features(df):
    df.drop('educacion_alcanzada', axis='columns', inplace = True)
    df.drop('barrio', axis='columns', inplace = True)
    return df
    
def missings_treatment(df):
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace(np.nan, 'sin_informar')
    df['trabajo'] = df['trabajo'].replace(np.nan, 'sin_informar')
    df.loc[df['categoria_de_trabajo'] == 'sin_trabajo', ['trabajo']] = 'sin_trabajo'
    return df

def one_hot_encodding(df):
    return pd.get_dummies(df, columns=['categoria_de_trabajo', 'estado_marital', 'genero', 'religion', 'rol_familiar_registrado', 'trabajo'], dummy_na=False, drop_first=True)

def dataset_split(X , y, test_size = 0.30):
    X_train, X_holdout, y_train, y_holdout =  train_test_split(
        X, 
        y, test_size = test_size, 
        random_state = 0, 
        stratify = y
    )
    X_train.reset_index(drop = True, inplace= True)
    X_holdout.reset_index(drop = True, inplace= True)
    y_train.reset_index(drop = True, inplace= True)
    y_holdout.reset_index(drop = True, inplace= True)
    return X_train, X_holdout, y_train, y_holdout


from sklearn.preprocessing import OrdinalEncoder
def ordinal_encode(df):
    categorias = [
     'preescolar',
     '1-4_grado',
     '5-6_grado',
     '7-8_grado',
     '9_grado',
     '1_anio',
     '2_anio',
     '3_anio',
     '4_anio',
     '5_anio',
     'universidad_1_anio',
     'universidad_2_anio',
     'universidad_3_anio',
     'universidad_4_anio',
     'universiada_5_anio',
     'universiada_6_anio'
    ]
    df_encoded = df
    oe = OrdinalEncoder(categories= [categorias])
    df_encoded[["educacion_alcanzada_encoded"]] = oe.fit_transform(df_encoded[["educacion_alcanzada"]])
    df_encoded = df_encoded.drop('educacion_alcanzada', axis=1)
    return df_encoded


def embedded(X, y, clf = DecisionTreeClassifier(random_state=117), min_importance=0.05):
    X_embedded = X
    feature_importance = 0
    while(feature_importance < min_importance):
        clf.fit(X, y)
        min_feature = np.argmin(clf.feature_importances_)
        feature_importance = clf.feature_importances_[min_feature]
        feature = X.columns[min_feature]
        X_embedded.drop(feature, axis=1, inplace=True)
    return X_embedded


def escalar(X_train, X_test):
    from sklearn.preprocessing import StandardScaler  
    StandardScaler().fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    return X_train, X_test