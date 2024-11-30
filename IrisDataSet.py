from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.metrics._scorer import average
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['label'] = iris.target
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print(iris_df.head())

    if True:
        sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                     'petal width (cm)', 'species']], hue='species')
        plt.show()

        #
        sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)', 'species']], hue='species')
        plt.show()

        #
        sns.pairplot(iris_df[['petal length (cm)', 'petal width (cm)', 'species']], hue='species')
        plt.show()

        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        sns.violinplot(x='species', y='petal length (cm)', data=iris_df)
        plt.subplot(2, 2, 2)
        sns.violinplot(x='species', y='petal width (cm)', data=iris_df)
        plt.subplot(2, 2, 3)
        sns.violinplot(x='species', y='sepal length (cm)', data=iris_df)
        plt.subplot(2, 2, 4)
        sns.violinplot(x='species', y='sepal width (cm)', data=iris_df)
        plt.show()

    #Separando os dados
    X = iris.data #Conjunto de dados do modelo
    y = iris.target #Coluna objetivo

    #Parâmetros da função de treino: conjunto de dados, coluna objetivo, Tamanho da amostra de teste (30% dos dados totais)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    #Modelo de classificação linear
    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)

    svm_predict = clf.predict(X_test)

    print(f'Teste executado em: {datetime.now()}')

    # Matriz de Confusão
    metrics_matriz_confusao = metrics.confusion_matrix(y_test, svm_predict)

    # Métricas (Avaliando se o modelo foi treinado corretamente
    metrics_acuracia = metrics.accuracy_score(y_test, svm_predict)
    metrics_precisao = metrics.precision_score(y_test, svm_predict, average='weighted')
    metrics_recall = metrics.recall_score(y_test, svm_predict, average='weighted')
    metrics_fscore = metrics.f1_score(y_test, svm_predict, average='weighted')

    print("Matriz: \n", metrics_matriz_confusao)

    print("Acurácia: ", metrics_acuracia)
    print("Precisão: ", metrics_precisao)
    print("Revocação: ", metrics_recall)
    print("F1-Score: ", metrics_fscore)