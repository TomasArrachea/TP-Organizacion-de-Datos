import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.metrics import classification_report, roc_curve, plot_confusion_matrix, auc
from sklearn.metrics import accuracy_score, precision_score, roc_curve, f1_score, recall_score


def plt_distribucion_de_clases(y):
    plt.style.use('ggplot')
    ax = pd.Series(y, name="label").value_counts(normalize=True).plot(kind='pie', autopct="%.2f%%")
    plt.title('Distribucion de clases en muestra')
    plt.show()
    
def plot_roc(modelo, X, y, classifier=True, y_proba=np.array(0)):
    if classifier == True:
         y_proba = modelo.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(15, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_matriz_de_confusion(modelo, X, y):
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.grid(False)
    plot_confusion_matrix(modelo, X, y, cmap=plt.cm.Blues, ax=ax)
    plt.title('Matriz de confusion')
    plt.show()
    
def imprimir_informe(modelo, X_holdout, y_holdout):
    y_pred = modelo.predict(X_holdout)

    plt_distribucion_de_clases(y_holdout)
    plot_matriz_de_confusion(modelo, X_holdout, y_holdout)
    plot_roc(modelo, X_holdout, y_holdout)

    print('Métricas:')
    print(classification_report(y_holdout, y_pred))
    
def imprimir_metricas(rscv, X, y, nombre):
    print(f"Resultados {nombre}")
    print(f"    Mejores hiperparámetros: {rscv.best_params_}")
    print(f"    Métrica AUC ROC: {rscv.best_score_:.2f}")
    print("    Otras metricas:")
    print(classification_report(y, rscv.best_estimator_.predict(X)))
       
def imprimir_metricas_red(y_true, y_pred):
    roc = roc_curve(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, pos_label=0)
    f1score = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy} \nPrecision: {precision} \nRecall: {recall} \nF1-score: {f1score} \n")

def get_df_test():
    GSPREADHSEET_DOWNLOAD_URL = (
        "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
    )
    FIUFIP_2021_1_GID = '1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE'
    df_test = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=FIUFIP_2021_1_GID))
    df_test.drop('representatividad_poblacional', axis=1, inplace=True)
    df_test.drop('id', axis=1, inplace=True)
    return df_test

def save_pred(y_pred, nombre):
    pd.DataFrame(y_pred, columns=['predictions']).to_csv(f'{nombre}.csv')