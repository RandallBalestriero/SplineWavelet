from numpy import *
import theano
import lasagne
from numpy.random import *

#############################################################################################################################################
#
#
#			SPLINE UTILITIES
#
#
#
#############################################################################################################################################


#utilise for linspace in theano, same behavior as with numpy
def th_linspace(start,end,n):
        n = theano.tensor.cast(n,'int32')
        a = theano.tensor.arange(n,dtype='float32')/theano.tensor.cast(n-1,'float32')
        a*=(end-start)
        return a+start


class theano_hermite_complex:
        def __init__(self,S,deterministic,initialization,renormalization,chirplet=0):
                """S: integer (the number of regions a.k.a piecewise polynomials)
                deterministic: bool (True if hyper-parameters are learnable)
                initialization: 'random','gabor','random_apodized' 
                renormalization: fn(x):norm(x) (theano function for filter renormalization)"""
                self.S               = S
		self.chirplet        = chirplet
                self.renormalization = renormalization
                self.deterministic   = deterministic
                T                    = theano.tensor.iscalar()# will be use to requiest a filter of length T
                self.mask            = ones(S,dtype='float32') # MASK will be used to apply boundary conditions
                self.mask[[0,-1]]    = 0 # boundary conditions correspond to 0 values on the boundaries
                if(initialization=='gabor'):
                        aa          = ones(S)
                        aa[::2]    -= 2
                        thetas_real = aa*hanning(S)**2
                        thetas_imag = roll(aa,1)*hanning(S)**2
                        gammas_real = zeros(S)
                        gammas_imag = zeros(S)
                        c           = zeros(1)
                elif(initialization=='random'):
                        thetas_real = rand(S)*2-1
                        thetas_imag = rand(S)*2-1
                        gammas_real = rand(S)*2-1
                        gammas_imag = rand(S)*2-1
                        if(chirplet):
                                c   = zeros(1)
                        else:
                                c   = zeros(1)
                elif(initialization=='random_apodized'):
                        thetas_real = (rand(S)*2-1)*hanning(S)**2
                        thetas_imag = (rand(S)*2-1)*hanning(S)**2
                        gammas_real = (rand(S)*2-1)*hanning(S)**2
                        gammas_imag = (rand(S)*2-1)*hanning(S)**2
                        if(chirplet):
                                c   = zeros(1)
                        else:
                                c   = zeros(1)
		if(chirplet):
                	self.c            = theano.shared(c.astype('float32'))
                self.thetas_real  = theano.shared(thetas_real.astype('float32'))
                self.thetas_imag  = theano.shared(thetas_imag.astype('float32'))
                self.gammas_real  = theano.shared(gammas_real.astype('float32'))
                self.gammas_imag  = theano.shared(gammas_imag.astype('float32'))
		# NOW CREATE THE POST PROCESSED VARIABLES
		self.ti           = th_linspace(0,1,self.S).dimshuffle([0,'x'])# THIS REPRESENTS THE MESH
                if(self.chirplet):
			thetas_real = self.thetas_real
                        thetas_imag = self.thetas_imag
                        gammas_real = self.gammas_real
                        gammas_imag = self.gammas_imag
			c = self.c
                        TT          = c.repeat(S)*float32(2*3.14159)*(th_linspace(0,1,S)**2)
			TTc         = c.repeat(S)*float32(2*3.14159)*th_linspace(0,1,S)
                        thetas_real = thetas_real*theano.tensor.cos(TT)-thetas_imag*theano.tensor.sin(TT)
                        thetas_imag = thetas_imag*theano.tensor.cos(TT)+thetas_real*theano.tensor.sin(TT)
			gammas_real = gammas_real*theano.tensor.cos(TT)-gammas_imag*theano.tensor.sin(TT)-thetas_real*theano.tensor.sin(TT)*TTc-thetas_imag*theano.tensor.cos(TT)*TTc
                        gammas_imag = gammas_imag*theano.tensor.cos(TT)+gammas_real*theano.tensor.sin(TT)+thetas_real*theano.tensor.cos(TT)*TTc-thetas_imag*theano.tensor.sin(TT)*TTc
		else:
			thetas_real = self.thetas_real
			thetas_imag = self.thetas_imag
			gammas_real = self.gammas_real
			gammas_imag  = self.gammas_imag
		#NOW APPLY BOUNDARY CONDITION
                self.pthetas_real   = ((thetas_real-thetas_real[1:-1].mean())*self.mask).dimshuffle([0,'x'])
                self.pthetas_imag   = ((thetas_imag-thetas_imag[1:-1].mean())*self.mask).dimshuffle([0,'x'])
                self.pgammas_real   = (gammas_real*self.mask).dimshuffle([0,'x'])
                self.pgammas_imag   = (gammas_imag*self.mask).dimshuffle([0,'x'])
        def get_filters(self,T):
                """method to obtain one filter with length T"""
		t               = th_linspace(0,1,T).dimshuffle(['x',0])#THIS REPRESENTS THE CONTINOUS TIME (sampled)
		#COMPUTE FILTERS BASED ONACCUMULATION OF WINDOWED INTERPOLATION
		real_filter     = self.interp((t-self.ti[:-1])/(self.ti[1:]-self.ti[:-1]),self.pthetas_real[:-1],self.pgammas_real[:-1],self.pthetas_real[1:],self.pgammas_real[1:]).sum(0)
                imag_filter     = self.interp((t-self.ti[:-1])/(self.ti[1:]-self.ti[:-1]),self.pthetas_imag[:-1],self.pgammas_imag[:-1],self.pthetas_imag[1:],self.pgammas_imag[1:]).sum(0)
		# RENORMALIZE
		K = self.renormalization(real_filter)+self.renormalization(imag_filter)
                real_filter     = real_filter/(K+0.00001)
                imag_filter     = imag_filter/(K+0.00001)
                return real_filter,imag_filter
        def interp(self,t,pi,mi,pip,mip):
                values = ((2*t**3-3*t**2+1)*pi+(t**3-2*t**2+t)*mi+(-2*t**3+3*t**2)*pip+(t**3-t**2)*mip)
                mask   = theano.tensor.cast(theano.tensor.ge(t,0),'float32')*theano.tensor.cast(theano.tensor.lt(t,1),'float32')
                return values*mask



def create_center_filter_complex(filter_length,max_length,class_function):
	deltaT = max_length-filter_length
	print deltaT
        real_f,imag_f =class_function(filter_length)
	filters = theano.tensor.concatenate([real_f.reshape((1,-1)),imag_f.reshape((1,-1))],axis=0)
#	if(deltaT!=0):
	return theano.tensor.roll(theano.tensor.concatenate([filters,theano.tensor.zeros((2,deltaT),dtype='float32')],axis=1),deltaT/2,axis=1)
#	else:
#		return filters
#        last_fb = theano.tensor.inc_subtensor(last_fb[2*row,deltaT/2:deltaT/2+filter_length],real_f)
#        last_fb = theano.tensor.inc_subtensor(last_fb[2*row+1,deltaT/2:deltaT/2+filter_length],imag_f)
#        return last_fb



def create_filter_banks_complex(filter_class,T,J,Q):
        scales = array(2**(arange(J*Q+1,dtype='float32')/Q)).astype('float32')#all the scales
        Ts     = array([T*scale for scale in scales]).astype('int32')#all the filter size support
#	Ts_t   = theano.shared(Ts)
	print "Lengths of the filters:",Ts
#	filters = [filter_class.get_filters(t) for t in Ts]
	filter_bank = theano.tensor.concatenate([create_center_filter_complex(i,Ts[-1],filter_class.get_filters) for i in Ts[:-1]],axis=0)
#        filter_bank,_=theano.scan(fn = lambda filter_length,max_length: create_center_filter_complex(filter_length,max_length,filter_class.get_filters),sequences=Ts[:-1],non_sequences=Ts[-1])
#	filter_bank=filter_bank[-1]
        return filter_bank[::2,:],filter_bank[1::2,:],Ts[-1]#filter_bank[::2],filter_bank[1::2],Ts[-1]


def create_center_filter_real(row,filter_length,last_fb,max_length,class_function):
	deltaT = max_length-filter_length
        real_f =class_function(filter_length)
        last_fb = theano.tensor.inc_subtensor(last_fb[row,deltaT/2:deltaT/2+filter_length],real_f)
        return last_fb



def create_filter_banks_real(filter_class,T,J,Q):
        scales = array(2**(arange(J*Q+1,dtype='float32')/Q)).astype('float32')#all the scales
        Ts     = array([T*scale for scale in scales]).astype('int32')#all the filter size support
	Ts_t   = theano.shared(Ts)
	print "Lengths of the filters:",Ts
        filter_bank,_=theano.scan(fn = lambda row,filter_length,last_fb,max_length: create_center_filter_real(row,filter_length,last_fb,max_length,filter_class.get_filters),sequences=[theano.tensor.arange(J*Q+1,dtype='int32'),Ts_t],non_sequences=[Ts_t[-1]],n_steps=J*Q+1,outputs_info=theano.tensor.zeros(((J*Q+1),Ts[-1]),dtype='float32'))
	filter_bank=filter_bank[-1]
        return filter_bank,Ts[-1]



class SplineFilter1D(lasagne.layers.Layer):
        """class wrapping the lasagne toolbox for use of the spline filters"""
        def __init__(self,incoming,N,J,Q,S,type_='hermite',stride=1,pad='valid',nonlinearity=1,deterministic=0,renormalization=theano.tensor.max,initialization='gabor',chirplet=0,complex_=1, **kwargs):
                """N:int, length of the smallest (highest frequency) filter >=S
                J: int, number of octave to decompose
                Q: int, number of filters per octave
                S: int, number of spline regions
                type_: 'hermite','lagrange'
                deterministic:bool, wether to update or not the hyper parameters for the splines
                normalization: theano function to renormalize each of the filters individually
                initialization:'gabor','random_random_apodized', initialization of the spline hyper parameters
                nonlinearity :bool either complex modulus or nothing, for now keep True"""
                super(SplineFilter1D, self).__init__(incoming, **kwargs)
                self.type_           = type_
                self.deterministic   = deterministic
                self.renormalization = renormalization
                self.initialization  = initialization
		self.chirplet        = chirplet
		self.complex         = complex_
		self.n_filters=J*Q
		if(complex_):
	                self.filter_class  = theano_hermite_complex(S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
       		        real_filter_bank_,imag_filter_bank_,T= create_filter_banks_complex(self.filter_class,N,J,Q)
       		        self.real_filter_bank      = theano.tensor.cast(real_filter_bank_.reshape((J*Q,1,T)),'float32')
       		        self.imag_filter_bank      = theano.tensor.cast(imag_filter_bank_.reshape((J*Q,1,T)),'float32')
			self.get_filters           = theano.function([],[self.real_filter_bank,self.imag_filter_bank])
       	       		self.real_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q),filter_size=int(T),W=self.real_filter_bank,stride=int(stride),pad=pad,nonlinearity=None,b=None)
               		self.imag_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q),filter_size=int(T),W=self.imag_filter_bank,stride=int(stride),pad=pad,nonlinearity=None,b=None)
		#	filters,Ts = create_filter_banks_complex(self.filter_class,N,J,Q)
		#	self.Ts=Ts
		#	self.filters_real = [f[0] for f in filters]
		#	self.filters_complex = [f[1] for f in filters]
		#	self.layers_real = [lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(1),filter_size=int(t),W=w.reshape((1,1,t)),stride=int(stride),pad='valid',nonlinearity=None,b=None) for w,t in zip(self.filters_real,Ts)]
                #        self.layers_complex = [lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(1),filter_size=int(t),W=w.reshape((1,1,t)),stride=int(stride),pad='valid',nonlinearity=None,b=None) for w,t in zip(self.filters_complex,Ts)]
		else:
                        self.filter_class  	   = theano_hermite_real(S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
                        real_filter_bank_,T	   = create_filter_banks_real(self.filter_class,N,J,Q)
                        self.real_filter_bank      = theano.tensor.cast(real_filter_bank_.reshape((J*Q+1,1,T)),'float32')
                        self.get_filters           = theano.function([],[self.real_filter_bank])
                        self.real_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q+1),filter_size=int(T),W=self.real_filter_bank,stride=int(stride),pad=int(pad),nonlinearity=None,b=None)


        def get_output_for(self, input, **kwargs):
		if(self.complex):
        	        real_output = self.real_layer.get_output_for(input)
        	        imag_output = self.imag_layer.get_output_for(input)
        	        return theano.tensor.abs_(real_output)+theano.tensor.abs_(imag_output)
		else:
                        real_output = self.real_layer.get_output_for(input)
                        return theano.tensor.abs_(real_output)
        def get_output_shape_for(self, input_shape):
                return self.real_layer.get_output_shape_for(input_shape)
        def get_params(self, **kwargs):
		if(self.complex):
			if(self.deterministic):
				return []
			if(self.chirplet):
				return [self.filter_class.thetas_real,self.filter_class.thetas_imag,self.filter_class.gammas_real,self.filter_class.gammas_imag,self.filter_class.c]
			else: 
				return [self.filter_class.thetas_real,self.filter_class.thetas_imag,self.filter_class.gammas_real,self.filter_class.gammas_imag]
		else:
                      if(self.deterministic):
                              return []
                      k= [self.filter_class.thetas_real,self.filter_class.gammas_real,self.filter_class.c]
                      if(self.chirplet): return k
                      else: return k[:-1]


def mad(x):
	median_ = theano.tensor.sort(x,1)[:,x.shape[1]/2]
	return median_/0.6741



class ExtremDenoising(lasagne.layers.Layer):
        def __init__(self,incoming,sigma_eval = mad,**kwargs):
                super(ExtremDenoising, self).__init__(incoming, **kwargs)
                W_real,W_imag = incoming.get_filters()
                W_real = W_real[:,0,:]
                W_imag = W_imag[:,0,:]
                self.sigma_eval = sigma_eval
                sigmas = theano.tensor.fvector()
                scalo = theano.tensor.fmatrix()
                self.corr_W_real = theano.tensor.dot(W_real,W_real.T)
                self.corr_W_imag = theano.tensor.dot(W_imag,W_imag.T)
                W = W_real+1j*W_imag
                #MM = theano.tensor.nlinalg.MatrixPinv()
                #M = MM(W)
		M = theano.tensor.transpose(W)
                M_real = theano.tensor.real(M)
                M_imag = theano.tensor.imag(M)
                self.corr_M_real=theano.tensor.dot(M_real.T,M_real)
                self.corr_M_imag=theano.tensor.dot(M_imag.T,M_imag)
        def get_output_for(self,scalos,**kwargs):
                def mini_denoising(sigma,xt,G,D):
                        xx     = xt.dimshuffle([0,'x'])*xt.dimshuffle(['x',0])
                        diracs = theano.tensor.gt((xx*G).sum(1),D.sum(1))
                        return diracs
                def apply_denoising(scalo,G,D):
                        sigmas = self.sigma_eval(scalo)
                        output,_ = theano.scan(fn=lambda xt,s,m,w:mini_denoising(s,xt,m,w),sequences=scalo.T,non_sequences=[sigmas,G,D])
                        return scalo*theano.gradient.disconnected_grad(output.T)
                G = theano.tensor.sqrt(self.corr_M_real**2+self.corr_M_imag**2)
                D = theano.tensor.sqrt((self.corr_M_real*self.corr_W_real-self.corr_M_imag*self.corr_W_imag)**2+(self.corr_W_real*self.corr_M_imag+self.corr_W_imag*self.corr_M_real)**2)
                output,_ = theano.scan(fn=apply_denoising,sequences=scalos,non_sequences=[G,D])   
                return output.dimshuffle([0,'x',1,2])
        def get_output_shape_for(self,input_shape):
                return (input_shape[0],1,input_shape[1],input_shape[2])
        def get_params(self,**kwargs):
                return []

