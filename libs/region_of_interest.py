class ROI:

	def __init__(self, image, position, number = None):
		self.image = image
		self.position = position
		self.number = number


	def getPosition(self):
		return self.position


	def getImage(self):
		return self.image


	def getNumber(self):
		return self.number


	def setNumber(self, number):
		self.number = number