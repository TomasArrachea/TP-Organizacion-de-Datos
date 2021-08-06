import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, OrdinalEncoder, StandardScaler  

## Funciones utilizadas para el preprocesamiento de cada modelo

def remove_irrelevant_features(df):
    df.drop('educacion_alcanzada', axis='columns', inplace = True)
    df.drop('barrio', axis='columns', inplace = True)
    return df    
    
def missings_treatment(df):
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace(np.nan, 'sin_informar')
    df['trabajo'] = df['trabajo'].replace(np.nan, 'sin_informar')
    df.loc[df['categoria_de_trabajo'] == 'sin_trabajo', ['trabajo']] = 'sin_trabajo'
    return df

def one_hot_encoding(df):
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

numerical_features = ['anios_estudiados', 'edad', 'ganancia_perdida_declarada_bolsa_argentina', 'horas_trabajo_registradas']

def escalar(df, scaler = None):
    df_features_numericos = df[numerical_features]

    if (scaler == None):
        scaler = StandardScaler()
        scaler.fit(df_features_numericos)
    
    features_escalados = pd.DataFrame(scaler.transform(df_features_numericos), columns = numerical_features, index=df.index)
    df = df.drop(numerical_features, axis= 'columns').join(features_escalados)    
    return df, scaler

def normalizar(df, normalizer = None):
    if (normalizer == None):
        normalizer = Normalizer().fit(df[numerical_features].T)
    features_normalizados = normalizer.transform(df[numerical_features].T).T
    features_normalizados = pd.DataFrame(features_normalizados, columns = numerical_features, index=df.index)
    df = df.drop(numerical_features, axis= 'columns').join(features_normalizados)
    return df, normalizer

def pca(df, n_components=0.90, pca = None):
    if (pca == None):
        pca = PCA(n_components=n_components)
        pca.fit(df)
    df = pca.transform(df)
    return df, pca