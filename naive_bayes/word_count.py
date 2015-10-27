#Description: Class for holding a word and a count



class word_count:

	def __init__(self, word, count=0):
		self.word = word
		self.count = count

	def __str__(self):
		result = word + "  "+ str(count)

		return result

	def increment(self, increment = 1):
		self.count = self.count + increment

		return self.count

	def get_count(self):
		
		return self.count