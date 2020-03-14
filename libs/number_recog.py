import cv2
import numpy as np
from os import path
from skimage.metrics import structural_similarity

from libs.region_of_interest import ROI
from libs.grid_estimator import GridEstimator

class NumberRecog:

	def __init__(self, image_path, estimateGrid = True, visualise = True):
		self.image = cv2.imread(image_path)
		self.visualise = visualise

		self.cleanImage()

		self.grid_mean_width = int(self.image.shape[0] / 9)
		self.grid_mean_height = int(self.image.shape[1] / 9)

		if estimateGrid:
			estimator = GridEstimator(self.image)
			self.grid_mean_width, self.grid_mean_height = estimator.estimateGridSize()


	def showImage(self, image):
		if self.visualise:
			cv2.imshow('image', image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


	def getNumberContours(self):
		# Prepare image
		blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
		thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		
		# Get the contours with a full tree hierarchy
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		items_seen = []
		contours_of_interest = []

		# Copy the image to display the contours
		contour_image = self.display_image.copy()

		# Loop the contours and hierarchy
		for i, contour in enumerate(contours):
			area = cv2.contourArea(contour)

			# Check area of the contour, skip big areas
			if area > ((self.image.shape[0] ** 2) / 81):
				continue

			# Fetch the parent of the item
			(_, _, _, parent) = hierarchy[0][i]

			# The 4, 6, 8, and 9 get smaller contours on the inside. Luckily the hierarchy
			# makes sure that the smaller contours are always a child of the contour that
			# we are interested in. Therefore, if we have already seen the parent of this
			# item we want to skip it
			if parent not in items_seen:
				if self.visualise:
					x, y, w, h = cv2.boundingRect(contour)
					cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

				contours_of_interest.append(contour)

			items_seen.append(i)

		if self.visualise:
			self.showImage(contour_image)

		return contours_of_interest


	# Called with y, x instead of x, y to match numpy matrix indexing
	def getGridLocation(self, y, x):
		y_pos = int((y / self.grid_mean_height).round())
		x_pos = int((x / self.grid_mean_width).round())
		
		return (y_pos, x_pos)


	def cleanImage(self):
		# Make uniform size
		image_resize = cv2.resize(self.image, (500, 500))
		
		# Get the edges of the image
		gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
		edged = cv2.Canny(gray, 50, 200, 255)

		# Detect the outer contour
		contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		# Crop out the outer padding of the image
		x, y, w, h = cv2.boundingRect(contours[0])
		image_cropped = image_resize[y:y + h, x:x + w]

		image_resize = cv2.resize(image_cropped, (500, 500))

		# Keep a version of the image for visualisation
		self.display_image = image_resize

		# Grayscale the image for further processing
		self.image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)


	def getRegions(self):
		number_contours = self.getNumberContours()

		rois = []

		for contour in number_contours:
			(x, y, w, h) = cv2.boundingRect(contour)
			
			# Crop out the regions of interest
			roi_img = self.image[y:y + h, x:x + w]
			grid_loc = self.getGridLocation(y, x)
			# self.showImage(roi_img)

			rois.append(ROI(roi_img, grid_loc))

		return rois


	def loadExamples(self):
		self.number_examples = []

		for i in range(1, 10):
			example_path = 'numbers/%i.png' % i

			if not path.exists(example_path):
				raise Exception('Example number file %s doesn\'t exist yet, make sure to generate these first.' % example_path)
			
			number_image = cv2.imread(example_path)
			number_image = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)

			self.number_examples.append(number_image)


	def matchToExample(self, item):
		scores = []

		for example in self.number_examples:
			if example.shape != item.shape:
				same_size = cv2.resize(item.copy(), (example.shape[1], example.shape[0]))
			else:
				same_size = item.copy()

			# recog.showImage(example)
			# recog.showImage(item)

			(score, diff) = structural_similarity(same_size, example, full = True)

			scores.append(score)

		return scores.index(max(scores)) + 1


	def appendNumbersToROI(self, rois):
		for i, roi in enumerate(rois):
			probable_number = self.matchToExample(roi.getImage())
			
			rois[i].setNumber(probable_number)


	def extract(self):
		rois = self.getRegions()

		self.loadExamples()
		self.appendNumbersToROI(rois)

		return rois