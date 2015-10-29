# Description: Python script to load and visualize What's Cooking Data
# Other sources: used lemmitization code from @DipayanSinhaRoy
# https://www.kaggle.com/dipayan/whats-cooking/whatscooking-python
#%pylab
import numpy as np
from time import time
import pandas as pd
import re
import colormaps as cmaps  # cmaps.parula (matlab), cmaps.viridis, inferno, magma, plasma 
import matplotlib.pyplot as plt 
# import matplotlib.patheffects as PathEffects

from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# from tsne import bh_sne

class DataLoader:
    """A class for loading and visualizing whatscooking data
        get_data() : get training data. 
    """

    def __init__(self,filename='train.json'):
        self.filename=filename

        # Read JSON data using pandas
        # columns are: id, cuisine, ingredients
        data  = pd.read_json(filename)
        
        # Label Encoders
        labels = LabelEncoder()
        labels.fit(data.cuisine)
        self.classes = labels.classes_
        self.label_encode = labels.fit()
        self.label_decode = labels.transform()        

        # Get numerical labels for ytrain 
        y_train = labels.transform(data.cuisine)

        # Vectorization of ingredients Using WordNet lemmatization & Tfid
        data['ingredients_clean_string'] = [' , '.join(z).strip() for z in data['ingredients']]  
        data['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in data['ingredients']]

        vectorizer  = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_df=0.57, analyzer='word', token_pattern=r'\w+')
        x_train     = vectorizer.fit_transform(data.ingredients_string).todense()
        ingred_dict = vectorizer.vocabulary_

        self.y_train = y_train
        self.x_train = x_train

    def get_data(self,full=False):
        if not(full):
            # limit training data: british, chinese & indian
            idx = np.logical_or(self.y_train==1, self.y_train==3)
            idx = np.logical_or(idx, self.y_train==2)
            idx = np.logical_or(idx, self.y_train==4)
            idx = np.logical_or(idx, self.y_train==5)
            idx = np.logical_or(idx, self.y_train==7)
            x_train = self.x_train[idx]
            y_train = self.y_train[idx]
            return x_train, y_train
        else: 
            return self.x_train, self.y_train
    
    def visualize(self,algo='pca'):
        if algo=='pca':                 
            # Visualize Using PCA 
            t0 = time()
            x_pca = PCA(n_components=2).fit_transform(self.x_train)
            t1 = time()
            print("PCA: %.2g sec" % (t1 - t0))
            figure()
            scatter(x_pca[:,0], x_pca[:,1], c=self.y_train, cmap=cmaps.parula)
            title('PCA Visualization of Cuisines')
            xlabel('Component 1')
            ylabel('Component 2')

        elif algo=='tsne':
            # Visualize Using t-SNE
            print("Computing t-SNE embedding")
            t0 = time()
            #tsne = TSNE(n_components=2, init='pca', random_state=0)
            #x_tsne = tsne.fit_transform(x_train)
            x_tsne = bh_sne(x_train)
            t1 = time()
            print("t-SNE: %.2g sec" % (t1 - t0))
            figure()
            scatter(x_tsne[:,0], x_tsne[:,1], c=y_train, cmap=cmaps.parula)
            title('t-SNE Visualization of Cuisines')
            xlabel('Component 1')
            ylabel('Component 2')

        else:
            print('unsuported algorithm')             



