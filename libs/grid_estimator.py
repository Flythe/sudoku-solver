import cv2
import numpy as np

class GridEstimator:

	def __init__(self, image):
		self.image = image


	def showImage(self, image):
		cv2.imshow('image', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def getStructure(self, image, size, direction = 'horizontal'):
		total = image.shape[0 if direction == 'vertical' else 1]
		direction_size = int(total / size)

		if direction == 'vertical':
			return cv2.getStructuringElement(cv2.MORPH_RECT, (1, direction_size))
		elif direction == 'horizontal':
			return cv2.getStructuringElement(cv2.MORPH_RECT, (direction_size, 1))


	def applyStructure(self, image, struct, fill = False):
		if fill:
			applied = cv2.dilate(image, struct)
			applied = cv2.erode(applied, struct)
		else:
			applied = cv2.erode(image, struct)
			applied = cv2.dilate(applied, struct)

		return applied


	def estimateGridSize(self):		
		gray = cv2.bitwise_not(self.image)
		bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2)
		# self.showImage(bw)

		# Fill out the image because the adaptive threshold misses some pixels
		horz_struct = self.getStructure(bw, 50, 'horizontal')
		refined_image = self.applyStructure(bw.copy(), horz_struct, True)

		vert_struct = self.getStructure(bw, 50, 'vertical')
		refined_image = self.applyStructure(refined_image, vert_struct, True)
		
		# Remove the numbers from the image
		horz_struct = self.getStructure(bw, 4, 'horizontal')
		horizontal = self.applyStructure(refined_image.copy(), horz_struct)

		vert_struct = self.getStructure(bw, 4, 'vertical')
		vertical = self.applyStructure(refined_image.copy(), vert_struct)

		merged = horizontal + vertical
		# self.showImage(merged)

		# Get the grid contours
		contours, hierarchy = cv2.findContours(merged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		counter = 0
		grid_width = []
		grid_height = []
		inner_width = []
		inner_height = []

		# Loop the contours and store their height and width
		for c in contours:
			area = cv2.contourArea(c)
			
			# Filter the outer contour
			if area > ((bw.shape[0] ** 2) / 2):
				continue

			counter = counter + 1

			(_, _, w, h) = cv2.boundingRect(c)

			inner_width.append(w)
			inner_height.append(h)

			if counter % 9 == 0:
				grid_width.append(inner_width)
				grid_height.append(inner_height)
				inner_width = []
				inner_height = []
		
		# Get the mean width and height
		grid_mean_width = np.median(np.median(grid_width, 0))
		grid_mean_height = np.median(np.median(grid_height, 0))

		return (grid_mean_width, grid_mean_height)