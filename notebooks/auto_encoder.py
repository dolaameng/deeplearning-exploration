import theano.tensor as T 
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from pylearn2.models.model import Model 
import numpy as np 

class AutoencoderCost(DefaultDataSpecsMixin, Cost):

	supervised = False

	def expr(self, model, data, **kwargs):
		space, source = self.get_data_specs(model)
		space.validate(data)

		X = data 
		Xhat = model.reconstruct(X)
		loss = -(X*T.log(Xhat) + (1-X)*T.log(1-Xhat)).sum(axis = 1)
		return loss.mean()

class Autoencoder(Model):

	def __init__(self, nvis, nhid):
		
		super(Autoencoder, self).__init__()

		self.nvis = nvis
		self.nhid = nhid

		W_value = np.random.uniform(size = (self.nvis, self.nhid))
		self.W = sharedX(W_value, "W")
		b_value = np.zeros(self.nhid)
		self.b = sharedX(b_value, "b")
		c_value = np.zeros(self.nvis)
		self.c = sharedX(c_value, 'c')
		self._params = [self.W, self.b, self.c]

		self.input_space = VectorSpace(dim = self.nvis)

	def reconstruct(self, X):
		h = T.tanh(T.dot(X, self.W) + self.b)
		return T.nnet.sigmoid(T.dot(h, self.W.T) + self.c)