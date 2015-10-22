#Description: Uses Naiive Bayes Method to calculate probability of a cuisine given a list of ingredients
#


import pandas as pd
import re
from word_count import *
from ingredients import *
import sys	
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import math



#Function: Clean_up
#
#Description: Makes a word lowercase and removes punctuation
#
def clean_up(word):
	word = word.lower()
	word = word.strip()
	
	return word

#Function: get_training_data
#
#Description: Parses training data and gets necessary counts
#
def get_training_data():
	cuisines = {}
	cuisine_counts= {}
	vocabulary = []

	data  = pd.read_json('train.json')

	num_entries = len(data.cuisine)

	for i in range(num_entries):

		#If the cuisine has not been encountered before, create a new dictionary and list entry for it
		if(not data.cuisine[i] in cuisines):
			ingredient_list = ingredients()
			cuisines[data.cuisine[i]]=ingredient_list
			cuisine_counts[data.cuisine[i]]=1

		#Otherwise update the number of occurences of this particular type of cuisine
		else:
			cuisine_counts[data.cuisine[i]]= cuisine_counts[data.cuisine[i]] +1

		#Iterate through all of the ingredients in this particular entry
		for ingredient in data['ingredients'][i]:
			#Attempt to normalize the ingredient (ex. Green Tomatos = green tomato)
			ingredient = clean_up(ingredient)
			ingredient = WordNetLemmatizer().lemmatize(ingredient)

			#If the ingredient is not yet in the vocabulary, add it to the vocabulary
			if(not ingredient in vocabulary):
				vocabulary.append(ingredient)

			#Add the ingredient to the ingredient list for this type of cuisine
			cuisines[data.cuisine[i]].add_ingredient(ingredient)

	return vocabulary, num_entries, cuisine_counts, cuisines

#Function: Calculate Probability
#
#Description: Determines the probability of input ingredients given cuisine using plus one smoothing and naiive bayes
#
def calculate_probability_ingredients_given_cuisine(input_ingredients, vocabulary, cuisine, cuisine_entries):

	print("\n\n")

	probability = 10000

	for ingredient in input_ingredients:

		entries_with_ingredient = cuisine.get_instances_of_ingredient(ingredient) + 1
		entries_for_cuisine = cuisine_entries + len(vocabulary)
		prob_ingredient = (entries_with_ingredient/entries_for_cuisine)

		print("			Ingredient:" + ingredient)

		print("				Entries in cuisine with this ingredient:	" + str(entries_with_ingredient))
		print("				Entries in cuisine:	" + str(entries_for_cuisine))

		print("				Probability:	" + str(entries_with_ingredient/entries_for_cuisine))
		probability = probability * prob_ingredient

	return probability

def get_args():
	args = sys.argv 
	return args[1:]

if __name__ == "__main__":
	input_ingredients = get_args()

	vocabulary, num_entries, cuisine_counts, cuisines = get_training_data()


	for cuisine_type in cuisine_counts.keys():

		print("Cuising Type:" + cuisine_type)

		print("		Entries of this Cuisine:	" + str(cuisine_counts[cuisine_type]))
		print("		Total Cuisine Entries:	" + str(num_entries))

		print("		Probability:	" + str(cuisine_counts[cuisine_type]/num_entries))
		
		log_prob_of_cuisine = cuisine_counts[cuisine_type]/num_entries
		log_prob_of_ingredients_given_cuisine = calculate_probability_ingredients_given_cuisine(input_ingredients, vocabulary, cuisines[cuisine_type], cuisine_counts[cuisine_type])
		prob_of_cuisine_given_ingredients = log_prob_of_ingredients_given_cuisine + log_prob_of_cuisine
		
	
