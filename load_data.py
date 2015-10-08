# Author: Justice Amoh
# Description: Python script to load and visualize What's Cooking Data

import pandas as pd
import re
import matplotlib.pyplot as plt 
import matplotlib.patheffects as PathEffects

from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Read JSON data using pandas
# columns are: id, cuisine, ingredients
data  = pd.read_json('train.json')

labels = LabelEncoder()
labels.fit(data.cuisine)
all_classes = labels.classes_
num_classes = len(all_classes)

# Get numerical labels for ytrain 
y_train = labels.transform(data.cuisine)


# Get vectorized words for xtrain - Using TfidfVectorizer
# all_ingred = [val for examples in data.ingredients for val in examples]
# vectorizer = TfidfVectorizer(min_df=1)
# x_train = vectorizer.fit_transform(all_ingred)


# Using WordNet lemmatizer
data['ingredients_clean_string'] = [' , '.join(z).strip() for z in data['ingredients']]  
data['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in data['ingredients']]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_df=0.57, analyzer='word', token_pattern=r'\w+')
x_train = vectorizer.fit_transform(data.ingredients_string).todense()
# vectorizer.get_feature_names()

# t-SNE Embedding of cuisine data
print("Computing t-SNE embedding")
tsne = TSNE(n_components=2, init='pca', random_state=0)
x_tsne = tsne.fit_transform(x_train)

