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