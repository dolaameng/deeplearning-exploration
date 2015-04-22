from sklearn.base import BaseEstimator
from pylearn2.config import yaml_parse
from pylearn2.datasets import DenseDesignMatrix
import shutil, os
from os import path
from time import strftime


class Pylearn2Model(BaseEstimator):
	def __init__(self, name, model_yaml, algorithm_yaml):
		self.name = name
		self.model_yaml = model_yaml
		self.algorithm_yaml = algorithm_yaml
		## create model folder to store data and learned model
		timestamp = strftime("%b%d%Y-%H%M%S")
		self.folder_path = path.abspath("%s-%s" % (self.name, timestamp))
		if path.exists(self.folder_path):
			raise RuntimeError("working folder %s exists" % self.folder_path)
		else:
			os.mkdir(self.folder_path)
	def fit(self, X, y = None, valid_X = None, valid_y = None):
		pass
	def predict(self, X):
		pass
	def clean(self):
		"""Clean everything, e.g., saved folders, data and etc
		"""
		shutil.rmtree(self.folder_path)
		print "deleted %s" % self.folder_path