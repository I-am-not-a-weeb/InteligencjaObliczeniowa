import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


iris = pd.read_csv('iris.csv')



def network(data,*layers):
    (train_set, test_set) = train_test_split(data.values, train_size=0.7, random_state=286315)
    train_inputs = train_set[:, 0:4]
    train_classes = train_set[:, 4]
    test_inputs = test_set[:, 0:4]
    test_classes = test_set[:, 4]
    
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=5000)
    mlp.fit(train_inputs, train_classes)
    
    pred = mlp.predict(test_inputs)
    correct = metrics.accuracy_score(test_classes, pred)
    print(layers)
    print(correct)

    
network(iris, 2)
network(iris, 3)
network(iris, 3, 3)
network(iris, 4, 4, 3, 2)