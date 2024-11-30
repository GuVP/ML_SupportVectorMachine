import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def oneshot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix) #Dummies - O que ele faz? Cria colunas de valor binária para cada valor possível na coluna em questão.
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df = df.copy()

    df = df.drop('ID', axis=1)

    df = oneshot_encode(
        df,
        {
            'EDUCATION': 'EDU',
            'MARRIAGE': 'MAR'
        }
    )

    y = df['default.payment.next.month'].copy() #Separação da coluna objetivo. É o valor dela que devemos prever.
    X = df.drop('default.payment.next.month', axis=1).copy() #Removendo a coluna alvo do conjunto de dados para predição

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    #Estudar FitTransform - O que de fato ele faz? Normalizada nos dados.
    ##https://praquemgostadetecnologia.com.br/quais-sao-as-diferencas-entre-fit-transform-fit_transforme-e-predict-no-sklearn/
    return X, y


if __name__=='__main__':
    data = pd.read_csv('Database\\UCI_Credit_Card.csv') #https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

    #Usado para ver metadados das colunas
    data.info()

    corr = data.corr() #Identifica a correlação entre as variáveis

    plt.figure(figsize=(18,18))
    sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
    plt.title('Mapa de Correlação')
    plt.show() #Exibe Mapa de correlação (Mapa de calor)

    X, y = preprocess_inputs(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=123) #Divide o conjunto de dados para teste

    #Define Modelos de Classificação
    C=1
    rbfCLF = svm.SVC(C=C, kernel='rbf')
    linearCLF = svm.SVC(C=C, kernel='linear')

    rbfCLF.fit(X_train, y_train) #Comando de treino a partir dos dados de treinamento
    y_pred = rbfCLF.predict(X_test) #Rodando a predição do algoritmo para o conjunto X de teste dado o treinamento acima

    print('A precisão do algoritmo RBF foi de: ')
    print("{:.2f}%".format(rbfCLF.score(X_test, y_test) * 100)) #Avaliação da Taxa de acerto
    print(classification_report(y_test, y_pred)) #Métricas de Avaliação (Utilização do conceito da matriz de confusão)

    linearCLF.fit(X_train, y_train) #Comando de treino a partir dos dados de treinamento
    y_pred = linearCLF.predict(X_test) #Rodando a predição do algoritmo para o conjunto X de teste dado o treinamento acima

    print('A precisão do algoritmo Linear foi de: ')
    print("{:.2f}%".format(linearCLF.score(X_test, y_test) * 100)) #Avaliação da Taxa de acerto
    print(classification_report(y_test, y_pred)) #Métricas de Avaliação (Utilização do conceito da matriz de confusão)
