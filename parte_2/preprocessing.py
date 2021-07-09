import pandas as pd
import numpy as np

## Funciones utilizadas para el preprocesamiento de cada modelo



#Convertir las variables categóricas en numéricas
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
    # se quitan features no utiles (segun análisis exploratorio tp1)
    df.drop('educacion_alcanzada', axis='columns', inplace = True)
    df.drop('barrio', axis='columns', inplace = True)
    return df
    
def missings_treatment(df):
    # tratamiento de missings (visto en tp1)
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace(np.nan, 'sin_informar')
    df['trabajo'] = df['trabajo'].replace(np.nan, 'sin_informar')
    df.loc[df['categoria_de_trabajo'] == 'sin_trabajo', ['trabajo']] = 'sin_trabajo'
    return df

def one_hot_encodding(df):
    return pd.get_dummies(df, columns=['categoria_de_trabajo', 'estado_marital', 'genero', 'religion', 'rol_familiar_registrado', 'trabajo'], dummy_na=False, drop_first=True)

def dataset_split(df):
    # separo en set de entrenamiento y set de validacion usando la biblioteca model_selection de sklearn
    return train_test_split(df.drop('tiene_alto_valor_adquisitivo', axis= 'columns'), df.tiene_alto_valor_adquisitivo, test_size = 0.30, random_state = 0)

#Convertir las variables ordinales en numéricas
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


#Seleccionar los features más importantes para el predictor
def embedded(X, y, clf, min_importance=0.05):
    X_embedded = X
    feature_importance = 0
    while(feature_importance < min_importance):
        clf.fit(X, y)
        min_feature = np.argmin(clf.feature_importances_)
        feature_importance = clf.feature_importances_[min_feature]
        feature = X.columns[min_feature]
        X_embedded.drop(feature, axis=1, inplace=True)
    return X_embedded
