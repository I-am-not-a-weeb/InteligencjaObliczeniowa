import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

diabetes = pd.read_csv('diabetes-1-1.csv')

def network(data,*layers):
    (train_set, test_set) = train_test_split(data.values, train_size=0.7, random_state=286315)
    train_inputs = train_set[:, 0:8]
    train_classes = train_set[:, 8]
    test_inputs = test_set[:, 0:8]
    test_classes = test_set[:, 8]
    
    mlp = MLPClassifier(activation="logistic",hidden_layer_sizes=layers, max_iter=500)
    mlp.fit(train_inputs, train_classes)
    
    pred = mlp.predict(test_inputs)
    correct = metrics.accuracy_score(test_classes, pred)
    print(metrics.confusion_matrix(test_classes, pred))
    print(layers)
    print(correct)
    

(train_set, test_set) = train_test_split(diabetes.values, train_size=0.7, random_state=286315)
train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]


knn3 = KNeighborsClassifier(n_neighbors=3).fit(train_inputs, train_classes)
pred3 = knn3.predict(test_inputs)

print("KNN3")

correct3 = metrics.accuracy_score(test_classes, pred3)
print(metrics.confusion_matrix(test_classes, pred3))
print(correct3)

print("relu 3,2")
network(diabetes, 3,2)

print("relu 3,3,2")
network(diabetes, 3,3,2)

print("relu 6,4,2")
network(diabetes, 6,4,2)

print("relu 8,7,4,2,2")
network(diabetes, 8,7,4,2,2)
