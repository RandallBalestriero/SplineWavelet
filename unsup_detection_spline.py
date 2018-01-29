from numpy import *
import theano
import lasagne
from utils_old import *
from theano.tensor.shared_randomstreams import RandomStreams




class SCALO:
        def __init__(self,x_shape,S,N,J,Q,deterministic=0,initialization='random',renormalization=theano.tensor.max,chirplet=1,complex_=1,log_=1):
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**J)
		layers.append(SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_='hermite',stride=1,pad='valid',nonlinearity=1,deterministic=deterministic,initialization=initialization,renormalization=renormalization,chirplet=chirplet,complex_=complex_))
#                layers.append(ExtremDenoising(layers[-1]))
		if log_:
			layers.append(lasagne.layers.NonlinearityLayer(layers[-1],nonlinearity=lambda x: theano.tensor.log(x+0.00001)))
		shape = lasagne.layers.get_output_shape(layers[-1])
		output = lasagne.layers.get_output(layers[-1])
		loss = theano.tensor.abs_(output).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
		params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
		learning_rate = theano.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
                self.train    = theano.function([x,learning_rate],loss,updates=updates)
		self.get_filters = lambda :layers[2].get_filters()
		self.get_repr    = theano.function([x],lasagne.layers.get_output(layers[2]))
