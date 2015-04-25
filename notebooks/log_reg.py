## Logistic Regression Model - HOMEMADE
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.model import Model
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
import numpy as np 

## The order of DefaultDataSpecsMixin and Cost matters!
class LogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    supervised = True ## specify supervised learning costs
    
    ## need model to map input to output, need data to test with target
    def expr(self, model, data, **kwargs): 
        space, source = self.get_data_specs(model) ## model's data specification
        space.validate(data) ## use model's vector space to validate data
        
        ## All X, y, yhat are theano variables
        X, y = data ## since it is supervised cost, we got both
        yhat = model.logistic_regression(X) ## call model to map X to yhat
        loss = -(y * T.log(yhat)).sum(axis = 1) ## rowwise selection
        return loss.mean() ## take the mean
    
class LogisticRegression(Model):
    def __init__(self, nvis, nclasses):
        super(LogisticRegression, self).__init__() ## call superclass Model constructure
        
        self.nvis = nvis ## standard name for ninputs
        self.nclasses = nclasses ## standard name for noutputs
        
        ## all model parameters are shared variable in Theano
        ## they usually come with an initialization and a name
        W_value = np.random.uniform(size = (self.nvis, self.nclasses))
        self.W = sharedX(W_value, "W")
        b_value = np.zeros(self.nclasses)
        self.b = sharedX(b_value, "b")
        ## all model parameters should be recorded in self._params
        ## for gradient calculation and house-keeping
        self._params = [self.W, self.b]
        
        ## construct the data specification
        self.input_space = VectorSpace(dim = self.nvis)
        self.output_space = VectorSpace(dim = self.nclasses)
        
    def logistic_regression(self, X):
        return T.nnet.softmax(T.dot(X, self.W) + self.b)
