import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


iris = datasets.load_iris()

print(iris.target)
X = iris.data[:, :2]  
Y = iris.target

target_names = iris.target_names



#######
# Zwykly
#####

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Original Data')
plt.legend()
plt.xlim(-3, 8)  
plt.ylim(-3, 5)
plt.tight_layout()
####
# Skalowanie po z-score
####



scaler_z_score = StandardScaler()
X_z_score = scaler_z_score.fit_transform(X)
plt.subplot(1, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_z_score[Y == i, 0], X_z_score[Y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(-3, 8)  
plt.ylim(-3, 5)
plt.title('Z-Score Scaled Data')

###
# Skalowanie po min_max
####

scaler_min_max = MinMaxScaler()
X_min_max = scaler_min_max.fit_transform(X)
plt.subplot(1, 3, 3)
for i, target_name in enumerate(target_names):
    plt.scatter(X_min_max[Y == i, 0], X_min_max[Y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(-3, 8)  
plt.ylim(-3, 5)
plt.title('Min-Max Scaled Data')


plt.legend()
plt.show()

print(X_min_max, X)