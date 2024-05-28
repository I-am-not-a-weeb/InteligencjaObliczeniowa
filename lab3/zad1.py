import pandas as pd
from sklearn.model_selection import train_test_split


iris = pd.read_csv('iris.csv')
(train_set, test_set) = train_test_split(iris.values, train_size=0.7, random_state=286315)

def classify_iris(sl, sw, pl, pw):
    if pw < 0.8:
        return "setosa"
    elif pl > 4.9:
        return "virginica"
    else: 
        return "versicolor"
len = test_set.shape[0]
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
correct = 0

for i in range(len):
    if classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3]) == test_classes[i]:
        correct += 1
        
print(correct)
print(correct/len*100, "%")

#print(train_set)