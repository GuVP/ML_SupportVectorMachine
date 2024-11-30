from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

def ViolinPlot(df, target, collumn):
    plt.subplot(2, 2, 1)
    sns.violinplot(x=target, y=collumn, data=df)
    plt.show()

def TrataValorMonetario(df, coluna):
    for i in range(0 ,len(df[coluna])):
        df[coluna].__setitem__(i, df[coluna][i].replace(',', ''))
        df[coluna].__setitem__(i, df[coluna][i].replace('R$', ''))

def TratamentoDados(df):
    print("Tamanho inicial do conjunto de dados: ", len(df))

    TrataValorMonetario(df, 'hoa')
    TrataValorMonetario(df, 'rent amount')
    TrataValorMonetario(df, 'property tax')
    TrataValorMonetario(df, 'total')

    for coluna in df.columns:
        try:
            if not (df[coluna].dtype == 'int64'):
                for i in range(0, len(df[coluna])):
                    if (df[coluna][i].__contains__('R$')):
                        df[coluna].__setitem__(i, df[coluna][i].replace('R$', ''))
                    elif (df[coluna][i].__contains__('Incluso')):
                        df[coluna].__setitem__(i, '0')
                    elif (df[coluna][i].__contains__('-')):
                        df[coluna].__setitem__(i, '0')
                    elif (df[coluna][i].__contains__('not acept') or df[coluna][i].__contains__('not furnished')):
                        df[coluna].__setitem__(i, '0')
                    elif (df[coluna][i].__contains__('acept') or df[coluna][i].__contains__('furnished')):
                        df[coluna].__setitem__(i, '1')
        except Exception as e:
            print("Erro em: ", coluna)
            print(repr(e))

    filtro = df['hoa'] != 'Sem info'

    return df[filtro]

if __name__ == '__main__':
    df = pd.read_csv('Database\\houses_to_rent.csv')
    df.info()

    df_tratado = TratamentoDados(df)

    if False:
        sns.pairplot(df[['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal',
                         'furniture', 'hoa', 'rent amount', 'property tax','fire insurance', 'total']], hue='furniture')
        plt.show()

    #Separando os dados
    x = df_tratado[['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal',
            'hoa', 'rent amount', 'property tax','fire insurance', 'total']]
    y = df_tratado[['furniture']].values.ravel()

    # Parâmetros da função de treino: conjunto de dados, coluna objetivo, Tamanho da amostra de teste (30% dos dados totais)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

    # Modelo de classificação linear
    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)

    svm_predict = clf.predict(X_test)

    print(f'Teste executado em: {datetime.now()}')

    # Matriz de Confusão
    metrics_matriz_confusao = metrics.confusion_matrix(y_test, svm_predict)

    # Métricas (Avaliando se o modelo foi treinado corretamente)
    metrics_acuracia = metrics.accuracy_score(y_test, svm_predict)
    metrics_precisao = metrics.precision_score(y_test, svm_predict, average='weighted')
    metrics_recall = metrics.recall_score(y_test, svm_predict, average='weighted')
    metrics_fscore = metrics.f1_score(y_test, svm_predict, average='weighted')

    print("Matriz: \n", metrics_matriz_confusao)

    print("Acurácia: ", metrics_acuracia)
    print("Precisão: ", metrics_precisao)
    print("Revocação: ", metrics_recall)
    print("F1-Score: ", metrics_fscore)

