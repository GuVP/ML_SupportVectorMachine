from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    paresOrdenadosExcel = pd.read_excel("Database\ParOrdenado.xlsx")
    paresOrdenadosExcel.dropna() #deleta valores nulos
    #Separador de Dados
    X = paresOrdenadosExcel.iloc[:, [0,1]].values
    y = paresOrdenadosExcel.iloc[:, 2].values

    sns.pairplot(paresOrdenadosExcel, hue='Acima')
    plt.show()

    #Criador dos conjuntos de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 13)

    #Mostra a distribuição dos dados
    #print(X_train.shape)
    #print(X_test.shape)

    clf = svm.SVC(C=1.0, kernel='linear')
    clf.fit(X_train, y_train)

    clf.predict(X_test)

    print('A precisão do algoritmo foi de: ')
    print(clf.score(X_test, y_test) * 100 )
