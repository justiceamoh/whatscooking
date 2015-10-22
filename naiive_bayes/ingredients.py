#Description: A class for managing ingredient_counts

from word_count import *

class ingredients:

	def __init__ (self):

		#Keeps track of the ingredients used in a cuisine and how many entries from the training data use the specific ingredient
		self.ingredient_list={}

	def add_ingredient(self, ingredient):
		if(ingredient in self.ingredient_list):
			self.ingredient_list[ingredient].increment()
		else:
			ingredient_count = word_count(ingredient, 1)
			self.ingredient_list[ingredient] = ingredient_count

		return 0

	def get_instances_of_ingredient(self, ingredient):
		if(ingredient in self.ingredient_list):
			return self.ingredient_list[ingredient].get_count()
		else:
			return 0

