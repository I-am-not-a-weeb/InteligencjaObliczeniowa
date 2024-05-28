import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


all_contents = ""

with open("article.txt", 'r', encoding='utf-8') as file:
            # Read the file's content and append to all_contents
    all_contents += file.read() + "\n"  # Add a newline to separate file contents

#tokenizacja

list_str = nltk.word_tokenize(all_contents)
print("Po tokenizacji:",len(list_str))

#Stop-words

arr = ["mr", ",", ".", "``", "-", "''", "'", "'s"]
stop_words = stopwords.words('english')
stop_words.extend(arr)

filtered_sentence = [w for w in list_str if not w.lower() in stop_words]

for w in list_str:
    if w not in stop_words:
        filtered_sentence.append(w)
 

print("Po stop-words: ", len(filtered_sentence))

# lematyzacja

lemmatizer = WordNetLemmatizer()

lematized = []

for w in filtered_sentence:
    lematized.append(lemmatizer.lemmatize(w,pos='v'))

print("Po lematyzacji: ",len(lematized))

processed_text = ' '.join(lematized)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform([processed_text])

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

word_counts = df.sum().sort_values(ascending=False)

top_10_words = word_counts.head(10)

plt.figure(figsize=(10, 6))
top_10_words.plot(kind='bar')
plt.xlabel('wyrazy')
plt.ylabel('liczba występowań')
plt.xticks(rotation=45)
plt.savefig("F.png")

