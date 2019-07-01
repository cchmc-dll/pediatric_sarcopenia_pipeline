# import the necessary packages
from unet3d_utils.utils import *

class Preprocessor3D:
	def __init__(self,imageshape=(128,128,32)):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.imageshape = imageshape

	def preprocess(self, image):

		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)