import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.metrics import classification_report, roc_curve, plot_confusion_matrix, auc

from matplotlib import pyplot as plt


def plt_distribucion_de_clases(y):
    plt.style.use('ggplot')
    ax = pd.Series(y, name="label").value_counts(normalize=True).plot(kind='pie', autopct="%.2f%%")
    plt.title('Distribucion de clases en muestra')
    plt.show()
    
def plot_roc(modelo, X, y):
    fpr, tpr, thresholds = roc_curve(y, modelo.predict_proba(X)[:,1])
    
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