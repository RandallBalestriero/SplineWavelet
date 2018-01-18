from numpy import *
import theano
import lasagne
from utils import *

class NN2:
        def __init__(self,x_shape,type_='none',S=16):
                N = 32 # length of high frequency filter
                J = 5  # number of octave
                Q = 8  # number of filters per octave
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                if(type_=='none'):
                        layers.append(lasagne.layers.Conv1DLayer(layers[-1],num_filters=J*Q,filter_size=int(N*2**((J*Q-1.0)/Q)),stride=1,pad=512,nonlinearity=theano.tensor.abs_))
                else:
                        layers.append(SplineFilter1D(layers[-1],N=N,J=J,Q=Q,S=S,type_=type_,stride=1,pad=512,nonlinearity=theano.tensor.abs_))
                layers.append(lasagne.layers.Pool1DLayer(layers[-1],8,mode='average_inc_pad'))
                S = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],([0],1,[1],[2])))
                layers.append(lasagne.layers.Conv2DLayer(layers[-1],num_filters=64,filter_size=(8,32)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(1,4)))
                layers.append(lasagne.layers.Conv2DLayer(layers[-1],num_filters=128,filter_size=(5,5)))
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],(2,32)))
                layers.append(lasagne.layers.DenseLayer(layers[-1],64))
                layers.append(lasagne.layers.DenseLayer(layers[-1],10,nonlinearity=theano.tensor.nnet.softmax))
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                for l in layers:
                        print(lasagne.layers.get_output_shape(l))
                #
                output        = lasagne.layers.get_output(layers[-1])
                params        = lasagne.layers.get_all_params(layers[-1])#.get_params()+layers[-2].get_params()+layers[-4].get_params()
                loss          = lasagne.objectives.categorical_crossentropy(output,y).mean()
                updates       = lasagne.updates.adam(loss,params,0.001)
                accu          = lasagne.objectives.categorical_accuracy(output,y).mean()
                # FUNC
#                self.get_filter_bank = theano.function([],filter_bank1)
                self.train      = theano.function([x,y],loss,updates=updates)
                self.test      = theano.function([x,y],accu)



class NN1:
         def __init__(self,x_shape,S,N,J,Q,type_='none',deterministic=0,initialization='random',renormalization=theano.tensor.max):
#                N = 32 # length of high frequency filter
#                J = 5  # number of octave
#                Q = 8  # number of filters per octave
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**((J*Q-1.0)/Q))
                if(type_=='none'):
                        layers.append(lasagne.layers.Conv1DLayer(layers[-1],num_filters=int(J*Q),filter_size=int(filter_size),stride=int(N/2),pad=int(filter_size/2),nonlinearity=theano.tensor.abs_))
                        self.get_filters = theano.function([],layers[-1].W)
                else:
                        layers.append(SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_=type_,stride=int(N/2),pad=int(filter_size/2),nonlinearity=1,deterministic=deterministic,initialization=initialization,renormalization=renormalization))
                        self.get_filters = theano.function([],layers[-1].get_filters())
                layers.append(lasagne.layers.Pool1DLayer(layers[-1],4,mode='average_inc_pad'))
                layers.append(lasagne.layers.Conv1DLayer(layers[-1],num_filters=64,filter_size=128,pad=64,stride=2))
                layers.append(lasagne.layers.Pool1DLayer(layers[-1],8))
                layers.append(lasagne.layers.DenseLayer(layers[-1],10,nonlinearity=theano.tensor.nnet.softmax))
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                for l in layers:
                        print(lasagne.layers.get_output_shape(l))
                #
                output        = lasagne.layers.get_output(layers[-1])
                params        = lasagne.layers.get_all_params(layers[-1])#.get_params()+layers[-2].get_params()+layers[-4].get_params()
                loss          = lasagne.objectives.categorical_crossentropy(output,y).mean()
                updates       = lasagne.updates.adam(loss,params,0.001)
                accu          = lasagne.objectives.categorical_accuracy(output,y).mean()
                # FUNC
#                self.get_filter_bank = theano.function([],filter_bank1)
                self.train      = theano.function([x,y],loss,updates=updates)
                self.test      = theano.function([x,y],accu)

class scalo:
         def __init__(self,x_shape,S,N,J,Q,type_='none',deterministic=0,initialization='random',renormalization=theano.tensor.max,chirplet=0):
                x             = theano.tensor.fmatrix('x')
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**J)
                if(type_=='none'):
                        layers.append(lasagne.layers.Conv1DLayer(layers[-1],num_filters=int(J*Q+1),filter_size=int(filter_size),stride=1,pad=int(filter_size/2),nonlinearity=theano.tensor.abs_))
                        self.get_filters = theano.function([],layers[-1].W)
                else:
                        layers.append(SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_=type_,stride=1,pad=int(filter_size/2),nonlinearity=1,deterministic=deterministic,initialization=initialization,renormalization=renormalization,chirplet=chirplet))
                        self.get_filters = theano.function([],layers[-1].get_filters())
		layers.append(lasagne.layers.Pool1DLayer(layers[-1],4,mode='average_inc_pad'))
                output        = lasagne.layers.get_output(layers[-1])
                params        = lasagne.layers.get_all_params(layers[-1])
                loss          = theano.tensor.abs_(output).mean()#+lasagne.regularization.apply_penalty(params,lasagne.regularization.l1)*0.01
                updates       = lasagne.updates.adam(loss,params,0.15)
                self.get_scalo= theano.function([x],output)
                self.train    = theano.function([x],loss,updates=updates)





class NN0:
        def __init__(self,x_shape,S,N,J,Q,type_='none',deterministic=0,initialization='random'):
#                N = 32 # length of high frequency filter
#                J = 5  # number of octave
#                Q = 8  # number of filters per octave
                x             = theano.tensor.fmatrix('x')
                y             = theano.tensor.ivector('y')
                layers        = [lasagne.layers.InputLayer(x_shape,x)]
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(x_shape[0],1,x_shape[1])))
                filter_size = int(N*2**((J*Q-1.0)/Q))
                if(type_=='none'):
                        layers.append(lasagne.layers.Conv1DLayer(layers[-1],num_filters=int(J*Q),filter_size=int(filter_size),stride=int(N/2),pad=int(filter_size/2),nonlinearity=theano.tensor.abs_))
                        self.get_filters = theano.function([],layers[-1].W)
                else:
                        layers.append(SplineFilter1D(layers[-1],N=int(N),J=int(J),Q=int(Q),S=int(S),type_=type_,stride=int(N/2),pad=int(filter_size/2),nonlinearity=1,deterministic=deterministic,initialization=initialization))
                        self.get_filters = theano.function([],layers[-1].get_filters())
			layers.append(ExtremDenoising(layers[-1]))
                layers.append(lasagne.layers.GlobalPoolLayer(layers[-1]))
                layers.append(lasagne.layers.DenseLayer(layers[-1],128,nonlinearity=theano.tensor.nnet.relu))
                layers.append(lasagne.layers.DenseLayer(layers[-1],10,nonlinearity=theano.tensor.nnet.softmax))
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                for l in layers:
                        print(lasagne.layers.get_output_shape(l))
                #
                output        = lasagne.layers.get_output(layers[-1])
                params        = lasagne.layers.get_all_params(layers[-1])#.get_params()+layers[-2].get_params()+layers[-4].get_params()
                loss          = lasagne.objectives.categorical_crossentropy(output,y).mean()
                updates       = lasagne.updates.adam(loss,params,0.002)
                accu          = lasagne.objectives.categorical_accuracy(output,y).mean()
                # FUNC
#                self.get_filter_bank = theano.function([],filter_bank1)
                self.train      = theano.function([x,y],loss,updates=updates)
                self.test      = theano.function([x,y],accu)










