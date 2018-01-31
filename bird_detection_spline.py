from numpy import *
import theano
import lasagne
from utils_old import *
from theano.tensor.shared_randomstreams import RandomStreams
from pylab import *


class RomainLayer(lasagne.layers.Layer):
	def __init__(self,incoming,**kwargs):
		super(RomainLayer,self).__init__(incoming,**kwargs)
		self.mask = ones(self.input_shape[2],dtype='float32')
		self.mask[0]=0
		self.mask[1]=0
		self.snrg = RandomStreams()
	def get_output_for(self,input,deterministic=False,**kwargs):
		input=input*self.mask.reshape((1,1,-1,1))
		if(deterministic==True):
			return input
		shift_temps = self.snrg.random_integers(low=-44000,high=44000)
		shift_freq  = self.snrg.random_integers(low=-1,high=1)
		return theano.tensor.roll(theano.tensor.roll(input,shift_temps,axis=3),shift_freq,axis=2)

class spline_BULBUL:
        def __init__(self,x_shape,S,N,J,Q,deterministic=0,initialization='random',renormalization=theano.tensor.max,chirplet=1,aug=1,complex_=1,log_=1):
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
		self.aug      = aug
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**J)
		layers.append(SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_='hermite',stride=1,pad='valid',nonlinearity=1,deterministic=deterministic,initialization=initialization,renormalization=renormalization,chirplet=chirplet,complex_=complex_))
#                layers.append(ExtremDenoising(layers[-1]))
		shape = lasagne.layers.get_output_shape(layers[-1])
		layers.append(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])))
#                layers.append(lasagne.layers.NonlinearityLayer(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]),nonlinearity=lasagne.nonlinearities.rectify))
		if aug:
			layers.append(RomainLayer(layers[-1]))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],stride=(1,2**9),pool_size=(1,1024),mode='average_inc_pad'))
                if log_:
                        layers.append(lasagne.layers.NonlinearityLayer(layers[-1],nonlinearity=lambda x: theano.tensor.log(x+0.001)))
                layers.append(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],256,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],32,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
		layers.append(lasagne.layers.DenseLayer(layers[-1],1,nonlinearity=lasagne.nonlinearities.sigmoid))
		output = lasagne.layers.get_output(layers[-1])
		output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
		loss = lasagne.objectives.binary_crossentropy(output,y).mean()
        	accu = lasagne.objectives.binary_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
		params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
		learning_rate = theano.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
		self.predict  = theano.function([x],output_test)
                self.train    = theano.function([x,y,learning_rate],loss,updates=updates)
                self.test     = theano.function([x,y],accu)
		self.get_filters = lambda:layers[2].get_filters()
		self.get_repr    = theano.function([x],lasagne.layers.get_output(layers[4]))

class conv_BULBUL:
        def __init__(self,x_shape,S,N,J,Q,aug=1,log_=1):
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
		self.aug      = aug
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**J)
                L1 = lasagne.layers.Conv1DLayer(layers[-1],num_filters=J*Q,filter_size=int(N*2**((J*Q+1)/Q)),W=A.get_filters()[0],nonlinearity=theano.tensor.abs_)
                L2 = lasagne.layers.Conv1DLayer(layers[-1],num_filters=J*Q,filter_size=int(N*2**((J*Q+1)/Q)),W=A.get_filters()[1],nonlinearity=theano.tensor.abs_)
                layers.append(lasagne.layers.ElemwiseMergeLayer([L1,L2],theano.tensor.add))
		shape = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],stride=(1,2**9),pool_size=(1,1024),mode='average_inc_pad'))
		if log_:
	                layers.append(lasagne.layers.NonlinearityLayer(layers[-1],nonlinearity=lambda x: theano.tensor.log(x+0.00001)))
                shape = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]))
		if aug:
                	layers.append(RomainLayer(layers[-1]))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],256,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],32,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DenseLayer(layers[-1],1,nonlinearity=lasagne.nonlinearities.sigmoid))
                output = lasagne.layers.get_output(layers[-1])
                output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
                loss = lasagne.objectives.binary_crossentropy(output,y).mean()
                accu = lasagne.objectives.binary_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
                learning_rate = theano.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
                self.predict  = theano.function([x],output_test)
                self.train    = theano.function([x,y,learning_rate],loss,updates=updates)
                self.test     = theano.function([x,y],accu)
                self.get_filters = theano.function([],[L1.W,L2.W])
                self.get_repr    = theano.function([x],lasagne.layers.get_output(layers[4]))



class BULBUL:
        def __init__(self,x_shape,S,N,J,Q,aug=1):
                x             = theano.tensor.ftensor3('x')
                y             = theano.tensor.ivector('y')
		self.aug      = aug 
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                shape = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.BatchNormLayer(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])),axes=[0,1,3]))
                if aug:
			layers.append(RomainLayer(layers[-1]))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],256,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],32,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DenseLayer(layers[-1],1,nonlinearity=lasagne.nonlinearities.sigmoid))
                output = lasagne.layers.get_output(layers[-1])
                output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
                loss = lasagne.objectives.binary_crossentropy(output,y).mean()
                accu = lasagne.objectives.binary_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
                learning_rate = theano.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
                self.predict  = theano.function([x],output_test)
                self.train    = theano.function([x,y,learning_rate],loss,updates=updates)
                self.test     = theano.function([x,y],accu)
                self.get_filters = lambda :[]
                self.get_repr    = lambda x:x




class GABOR_BULBUL:
        def __init__(self,x_shape,S,N,J,Q,aug=1,log_=1):
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
                self.aug      = aug
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**J)
                A = SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_='hermite',stride=1,pad='valid',nonlinearity=1,deterministic=0,initialization='gabor',renormalization=lambda x:x.norm(2),chirplet=0,complex_=1)
                L1 = lasagne.layers.Conv1DLayer(layers[-1],num_filters=J*Q,filter_size=int(N*2**((J*Q+1)/Q)),W=A.get_filters()[0],nonlinearity=theano.tensor.abs_)
                L2 = lasagne.layers.Conv1DLayer(layers[-1],num_filters=J*Q,filter_size=int(N*2**((J*Q+1)/Q)),W=A.get_filters()[1],nonlinearity=theano.tensor.abs_)
                layers.append(lasagne.layers.ElemwiseMergeLayer([L1,L2],theano.tensor.add))
                shape = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],stride=(1,2**9),pool_size=(1,1024),mode='average_inc_pad'))
                if log_:
                        layers.append(lasagne.layers.NonlinearityLayer(layers[-1],nonlinearity=lambda x: theano.tensor.log(x+0.00001)))
                layers.append(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]))
                if aug:
                        layers.append(RomainLayer(layers[-1]))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(3,3)))
                layers.append(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(layers[-1],num_filters=16,filter_size=(1,3),nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,3)))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],256,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DropoutLayer(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layers[-1],32,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers.append(lasagne.layers.DenseLayer(layers[-1],1,nonlinearity=lasagne.nonlinearities.sigmoid))
                output = lasagne.layers.get_output(layers[-1])
                output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
                loss = lasagne.objectives.binary_crossentropy(output,y).mean()
                accu = lasagne.objectives.binary_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
                learning_rate = theano.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
                self.predict  = theano.function([x],output_test)
                self.train    = theano.function([x,y,learning_rate],loss,updates=updates)
                self.test     = theano.function([x,y],accu)
                self.get_filters = theano.function([],[L1.W,L2.W])
                self.get_repr    = theano.function([x],lasagne.layers.get_output(layers[4]))




