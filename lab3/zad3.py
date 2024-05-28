import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

iris = pd.read_csv('iris.csv')

(train_set, test_set) = train_test_split(iris.values, train_size=0.7, random_state=286315)
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

knn3 = KNeighborsClassifier(n_neighbors=3).fit(train_inputs, train_classes)
pred3 = knn3.predict(test_inputs)
correct3 = metrics.accuracy_score(test_classes, pred3)
print(correct3)

knn5 = KNeighborsClassifier(n_neighbors=5).fit(train_inputs, train_classes)
pred5 = knn5.predict(test_inputs)
correct5 = metrics.accuracy_score(test_classes, pred5)
print(correct5)

knn11 = KNeighborsClassifier(n_neighbors=11).fit(train_inputs, train_classes)
pred11 = knn11.predict(test_inputs)
correct11 = metrics.accuracy_score(test_classes, pred11)
print(correct11)

dd = tree.DecisionTreeClassifier().fit(train_inputs, train_classes)
predd = dd.predict(test_inputs)
correctdd = metrics.accuracy_score(test_classes, predd)
print(correctdd)

nb = GaussianNB().fit(train_inputs, train_classes)
prednb = nb.predict(test_inputs)
correctnb = metrics.accuracy_score(test_classes, prednb)
print(correctnb)