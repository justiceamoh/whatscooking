# Author: Justice Amoh
# Description: Python script to load and visualize What's Cooking Data

import pandas as pd
import matplotlib.pyplot as plt 


# Read JSON data using pandas
# columns are: cuisine, id, ingredients
train = pd.read_json('train.json')
all_classes = train.cuisine.unique() 
num_classes = len(all_classes)