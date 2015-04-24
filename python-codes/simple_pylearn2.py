
from pylearn2.config import yaml_parse
from pylearn2.datasets import DenseDesignMatrix
import cPickle

### HELPER FUNCTION TO CREATE YAML STRING ####################

def yamlize_densedesignmatrix(fname, X, y = None, alias = None):
	"""
	convert X, y to pylearn2.datasets.DenseDesignMatrix and pickle it 
	fname: the file path to pickle the data 
	X, y: np.array, must be dense (not scipy.Sparse)
	alias: alias used in yaml, e.g, 'train_data'
	RETURN yaml representation of the pickled data 
	"""
	data = DenseDesignMatrix(X = X, y = y)
	cPickle.dump(data, open(fname, 'w'))
	alias = "&%s " % alias if alias else ""
	return r"""%s!pkl: '%s'""" % (alias, fname)

def yamlize_transformerdataset():
	pass