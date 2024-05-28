import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv', sep=',')

items = set()
for col in df:
    items.update(df[col].unique())

itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row) 
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)



freq_items = apriori(ohe_df, min_support=0.005, use_colnames=True, verbose=1)
print("Apriori:",freq_items.head(7))

rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)
print("Assoc rules:",rules.head(40))

sorted_rules = rules.sort_values(by='confidence', ascending=False)
print("Sorted rules:",sorted_rules.to_string())

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.savefig("support-vs-confidence.png")

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.savefig("support-vs-lift.png")

plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('Lift vs Confidence')
plt.savefig("lift-vs-confidence.png")