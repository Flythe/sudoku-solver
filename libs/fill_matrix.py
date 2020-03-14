import numpy as np

def getFilledMatrix(rois):
	matrix = np.zeros((9, 9))

	for roi in rois:
		matrix[roi.getPosition()] = roi.getNumber()

	return matrix