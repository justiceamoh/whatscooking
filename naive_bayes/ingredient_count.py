#Description: Class for holding an ingredient and a count



class ingrd_count:

	def __init__(self, ingredient, count=0):
		self.ingredient = ingredient
		self.count = count

	def __str__(self):
		result = ingredient + "  "+ str(count)

		return result

	def increment(self, increment = 1):
		self.count = self.count + increment

		return self.count

	def get_count(self):
		
		return self.count