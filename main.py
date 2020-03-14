import cv2
from libs.number_recog import NumberRecog
from libs.fill_matrix import getFilledMatrix

recog = NumberRecog('puzzles/1.png')
number_extracts = recog.extract()

matrix = getFilledMatrix(number_extracts)

print(matrix)