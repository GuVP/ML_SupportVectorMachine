from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    produtoExcel = pd.read_excel('Database\ProdutoParOrdenado.xlsx')
    produtoExcel.dropna()

    X = produtoExcel.iloc[:, [0,1]].values
    y = produtoExcel.iloc[:, 2].values

    sns.pairplot(produtoExcel, hue='Acima')
    #plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

    clf = svm.SVC(C=100, kernel='rbf')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    print(clf.score(X_test, y_test))

    #plt.show()


