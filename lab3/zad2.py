import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = pd.read_csv('iris.csv')

(train_set, test_set) = train_test_split(iris.values, train_size=0.7, random_state=286315)

train_inputs = train_set[:, 0:4].tolist()
train_classes = train_set[:, 4].tolist()
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

iris_tree = tree.DecisionTreeClassifier()

iris_tree.fit(train_inputs, train_classes)

correct = 0

arr_test = iris_tree.predict(test_inputs)
for i in range(len(test_inputs)):
    if arr_test[i] == test_classes[i]:
        correct += 1
        
print(correct)
print(correct/len(test_inputs)*100, "%")