import pandas as pd
import random

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt

NA_values = ["NA", "na", "n/a", "missing", "MISSING", "MISSING DATA", "missing data", "N/A","-","?","??","???","????"]

iris = pd.read_csv("iris-1.csv")

iris_err = pd.read_csv("iris_with_errors.csv",na_values=NA_values)

#a) Policz ile jest w bazie brakujących lub nieuzupełnionych danych. Wyświetl statystyki bazy danych z błędami.
#print("NA values:\n",iris_err.isnull().sum())
#print("Statistics:\n",iris_err.describe())

num_col = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

#b) Sprawdź czy wszystkie dane numeryczne są z zakresu (0; 15). Dane spoza zakresu muszą być poprawione. Możesz
#tutaj użyć metody: za błędne dane podstaw średnią (lub medianę) z danej kolumny.

less_than_0 = iris_err[(iris_err[num_col] < 0).any(axis=1)]
more_than_15 = iris_err[(iris_err[num_col] > 15).any(axis=1)]
not_number = iris_err[iris_err.isna().any(axis=1)]


#print("<0",less_than_0)
#print(">15",more_than_15)
#print("nan",not_number)

for col in num_col:
    avg = iris_err[col].mean()
    iris_err[col].fillna(avg, inplace=True)
    iris_err[col] = iris_err[col].clip(lower=0, upper=15)


#c) Sprawdź czy wszystkie gatunki są napisami: „Setosa”, „Versicolor” lub „Virginica”. Jeśli nie, wskaż jakie popełniono
#błędy i popraw je własną (sensowną) metodą.


strs = ["Setosa", "Versicolor", "Virginica"]

for elem in iris_err["variety"]:
    if elem not in strs:
        elem = random.choice(strs)
        


# Zad 2

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.Series(iris.target, name='FlowerType')
print(X.head())
pca_iris = PCA(n_components=3).fit(iris.data)
print(pca_iris)
print(pca_iris.explained_variance_ratio_)
print(pca_iris.components_)
print(pca_iris.transform(iris.data))

print("EVR:", pca_iris.explained_variance_ratio_)

transformed_data = pca_iris.transform(iris.data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']

for i, flower_type in enumerate(iris.target_names):
    ax.scatter(transformed_data[Y == i, 0], transformed_data[Y == i, 1], transformed_data[Y == i, 2],
               c=colors[i],
               marker=markers[i],
               label=flower_type)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA Irysow')

plt.legend()
plt.show()
