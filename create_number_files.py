import cv2
from libs.number_recog import NumberRecog

recog = NumberRecog('puzzles/example.png')
number_extracts = recog.getRegions()

for extract in number_extracts:
	path = 'numbers/%i.png' % (extract.getPosition()[1] + 1)

	cv2.imwrite(path, extract.getImage())