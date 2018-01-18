from numpy import *
import theano
import lasagne
from numpy.random import *
import glob

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


def split_file_name(files,ratio):
	p           = permutation(len(files))
	cutoff      = int(len(files)*ratio)
	print cutoff
	train_files = [files[i] for i in p[:cutoff]]
	test_files  = [files[i] for i in p[cutoff:]]
	return train_files,test_files

def file_to_array(name,n):
	x = AudioSegment.from_file(name,"au").get_array_of_samples()#f.readframes(n), dtype=int16)
        x=asarray(x).astype('float32')
	i = randint(30000,len(x)-300000-n-1)
	x = x[i:i+n]
#	x = (x[::2]+x[1::2])/2
	x-=x.mean()
	x/=x.max()
	return x

def batch_to_array(x,names,n):
        for i,f in zip(range(len(names)),names):
                x[i]=file_to_array(f,n)


class GTZAN_generator:
	def __init__(self,batch_size,path = '../../DATASET/genres/',split_ratio = 0.7,n=661000):
		self.batch_size = batch_size 
		self.genres = sort(glob.glob(path+'*'))
		files_per_genre    = [glob.glob(genre+'/*') for genre in self.genres]
		self.train_files   = []
		self.test_files    = []
		self.n_train_files = 0
		self.n_test_files  = 0
		self.n = n
		for genre in self.genres:
			files = glob.glob(genre+'/*')
			train_files,test_files = split_file_name(files,split_ratio)
			self.n_train_files+=len(train_files)
			self.n_test_files+=len(test_files)
			self.train_files.append(train_files)
			self.test_files.append(test_files)
		self.n_train_batchs = self.n_train_files/batch_size
		self.n_test_batchs = self.n_test_files/batch_size
		print "GTZAN Generator:"
		print "\tNumber of Genres: ",len(self.genres)
		print "\tNumber of batchs: ",self.n_train_batchs,self.n_test_batchs
		self.current_train_batch_index = 0
		self.current_test_batch_index = 0
		self.g_files_per_batch = self.batch_size/len(self.genres)
		self.y = repeat(range(len(self.genres)),self.g_files_per_batch).astype('int32')
		self.x = zeros((len(self.y),n),dtype='float32')
	def train_batch(self):
		print "TRAIN BATCH"
		indices = range(self.current_train_batch_index*self.g_files_per_batch,(self.current_train_batch_index+1)*self.g_files_per_batch)
		for g in xrange(len(self.genres)):
			batch_to_array(self.x[g*self.g_files_per_batch:(g+1)*self.g_files_per_batch],[self.train_files[g][i] for i in indices],self.n)
		self.current_train_batch_index +=1
		return self.x,self.y
        def test_batch(self):
		print "TEXT BATCH"
                indices = range(self.current_test_batch_index*self.g_files_per_batch,(self.current_test_batch_index+1)*self.g_files_per_batch)
                for g in xrange(len(self.genres)):
                        batch_to_array(self.x[g*self.g_files_per_batch:(g+1)*self.g_files_per_batch],[self.test_files[g][i] for i in indices],self.n)
                self.current_test_batch_index +=1
                return self.x,self.y
	def train_epoch_done(self):
		return self.current_train_batch_index==self.n_train_batchs
	def test_epoch_done(self):
                return self.current_test_batch_index==self.n_test_batchs



class theano_lagrange:
        def __init__(self,S,deterministic,initialization,renormalization):
                """S: integer (the number of regions a.k.a piecewise polynomials)
                deterministic: bool (True if hyper-parameters are learnable)
                initialization: 'random','gabor','random_apodized' 
                renormalization: fn(x):norm(x) (theano function for filter renormalization)"""
                self.S               = S
                self.renormalization = renormalization
                self.deterministic   = deterministic
                self.mask            = ones(S,dtype='float32') # MASK will be used to apply boundary conditions
                self.mask[[0,-1]]    = 0 # boundary conditions correspond to 0 values on the boundaries
                if(initialization=='gabor'):
                        aa          = ones(S) 
                        aa[::2]    -= 2
                        thetas_real = aa*hanning(S)
                        thetas_imag = roll(thetas_real,1)
                        c           = zeros(1)
                elif(initialization=='random'):
                        thetas_real = randn(S)
                        thetas_imag = randn(S)
                        c           = randn(1)
                elif(initialization=='random_apodized'):
                        thetas_real = randn(S)*hanning(S)
                        thetas_imag = randn(S)*hanning(S)
                        c           = randn(1)
                self.thetas_real  = theano.shared(thetas_real.astype('float32'))
                self.thetas_imag  = theano.shared(thetas_imag.astype('float32'))
                self.c            = theano.shared(c)
        def get_real_filter(self,T):
                """method to obtain one filter with length T"""
                T               = theano.tensor.cast(T,'int32')
                eps             = theano.tensor.cast(1.0/T,'float32')
                f_t,updates     = theano.scan(fn = self.interp,non_sequences=[self.thetas_real*self.mask,th_linspace(0+eps,1-eps,theano.tensor.cast(theano.tensor.ceil(T/(self.S-1.0)),'float32')),T,self.S-1],sequences=[theano.tensor.arange(self.S-1,dtype='int32')],outputs_info=[theano.tensor.zeros((T,),dtype='float32')])
                filter_         = f_t[-1]*theano.tensor.cos(self.c.repeat(T)*theano.tensor.cast(3.14159*th_linspace(0,1,T)**2,'float32'))
                centered_filter = filter_-filter_.mean()
                final_filter    = centered_filter/self.renormalization(centered_filter)
                return final_filter
        def get_imag_filter(self,T):
                """method to obtain one filter with length T"""
                T               = theano.tensor.cast(T,'int32')
                eps             = theano.tensor.cast(1.0/T,'float32')
                f_t,updates     = theano.scan(fn = self.interp,non_sequences=[self.thetas_imag*self.mask,th_linspace(0+eps,1-eps,theano.tensor.cast(theano.tensor.ceil(T/(self.S-1.0)),'float32')),T,self.S-1],sequences=[theano.tensor.arange(self.S-1,dtype='int32')],outputs_info=[theano.tensor.zeros((T,),dtype='float32')])
                filter_         = f_t[-1]*theano.tensor.sin(self.c.repeat(T)*theano.tensor.cast(3.14159*th_linspace(0,1,T)**2,'float32'))
                centered_filter = filter_-filter_.mean()
                final_filter    = centered_filter/self.renormalization(centered_filter)
                return final_filter
        def interp(self,t,mi,pi,mip,pip):
		values = ((2*t**3-3*t**2+1)*pi+(t**3-2*t**2+t)*mi+(-2*t**3+3*t**2)*pip+(t**3-t**2)*mip)
		mask   = theano.tensor.cast(theano.tensor.ge(t,0),'float32')*theano.tensor.cast(theano.tensor.lt(t,1),'float32')
		return values*mask


class theano_hermite:
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
                        thetas_real = aa*hanning(S)
                        thetas_imag = roll(thetas_real,1)
                        gammas_real = zeros(S)
                        gammas_imag = zeros(S)
                        c           = zeros(1)
                elif(initialization=='random'):
                        thetas_real = randn(S)
                        thetas_imag = randn(S)
                        gammas_real = randn(S)
                        gammas_imag = randn(S)
                        c           = randn(1)
                elif(initialization=='random_apodized'):
                        thetas_real = randn(S)*hanning(S)
                        thetas_imag = randn(S)*hanning(S)
                        gammas_real = randn(S)*hanning(S)
                        gammas_imag = randn(S)*hanning(S)
                        c           = randn(1)
                self.c            = theano.shared(c.astype('float32'))
                self.thetas_real  = theano.shared(thetas_real.astype('float32'))
                self.thetas_imag  = theano.shared(thetas_imag.astype('float32'))
                self.gammas_real  = theano.shared(gammas_real.astype('float32'))
                self.gammas_imag  = theano.shared(gammas_imag.astype('float32'))
        def get_filters(self,T):
                """method to obtain one filter with length T"""
		ti              = th_linspace(0,1,self.S).dimshuffle([0,'x'])# THIS REPRESENTS THE MESH
		t               = th_linspace(0,1,T).dimshuffle(['x',0])#THIS REPRESENTS THE CONTINOUS TIME
		# APPLY CONSTRAINTS OF MEAN 0 AND BOUNDARY CONDITIONS
		thetas_real     = ((self.thetas_real-self.thetas_real[1:-1].mean())*self.mask).dimshuffle([0,'x'])
                thetas_imag     = ((self.thetas_imag-self.thetas_imag[1:-1].mean())*self.mask).dimshuffle([0,'x'])
		gammas_real     = (self.gammas_real*self.mask).dimshuffle([0,'x'])
                gammas_imag     = (self.gammas_imag*self.mask).dimshuffle([0,'x'])
		#COMPUTE FILTERS BASED ONACCUMULATION OF WINDOWED INTERPOLATION
		real_filter     = self.interp((t-ti[:-1])/(ti[1:]-ti[:-1]),thetas_real[:-1],gammas_real[:-1],thetas_real[1:],gammas_real[1:]).sum(0)
                imag_filter     = self.interp((t-ti[:-1])/(ti[1:]-ti[:-1]),thetas_imag[:-1],gammas_imag[:-1],thetas_imag[1:],gammas_imag[1:]).sum(0)
		# RENORMALIZE
                real_filter     = (real_filter)/self.renormalization(real_filter)
                imag_filter     = (imag_filter)/self.renormalization(imag_filter)
		#APPLY CHIRPLET
		if(self.chirplet):
			TT              = self.c.repeat(T)*float32(3.14159)*th_linspace(0,1,T)**2
			real_filter     = real_filter*theano.tensor.cos(TT)-imag_filter*theano.tensor.sin(TT)
                	imag_filter     = real_filter*theano.tensor.sin(TT)+imag_filter*theano.tensor.cos(TT)
                return real_filter,imag_filter
        def interp(self,t,pi,mi,pip,mip):
                values = ((2*t**3-3*t**2+1)*pi+(t**3-2*t**2+t)*mi+(-2*t**3+3*t**2)*pip+(t**3-t**2)*mip)
                mask   = theano.tensor.cast(theano.tensor.ge(t,0),'float32')*theano.tensor.cast(theano.tensor.lt(t,1),'float32')
                return values*mask






class filter_bank:
        def __init__(self,N,S,deterministic,initialization,renormalization,chirplet=0):
                """S: integer (the number of regions a.k.a piecewise polynomials)
                deterministic: bool (True if hyper-parameters are learnable)
                initialization: 'random','gabor','random_apodized' 
                renormalization: fn(x):norm(x) (theano function for filter renormalization)"""
                self.S               = S
		self.N = N
		self.chirplet        = chirplet
                self.renormalization = renormalization
                self.deterministic   = deterministic
                self.mask            = ones(S,dtype='float32') # MASK will be used to apply boundary conditions
                self.mask[[0,-1]]    = 0 # boundary conditions correspond to 0 values on the boundaries
                if(initialization=='gabor'):
                        aa          = ones(S)
                        aa[::2]    -= 2
                        thetas_real = aa*hanning(S)
                        thetas_imag = roll(thetas_real,1)
                        gammas_real = zeros(S)
                        gammas_imag = zeros(S)
                        c           = zeros(1)
                elif(initialization=='random'):
                        thetas_real = randn(N,S)
                        thetas_imag = randn(N,S)
                        gammas_real = randn(N,S)
                        gammas_imag = randn(N,S)
                        c           = randn(N)
                elif(initialization=='random_apodized'):
                        thetas_real = randn(S)*hanning(S)
                        thetas_imag = randn(S)*hanning(S)
                        gammas_real = randn(S)*hanning(S)
                        gammas_imag = randn(S)*hanning(S)
                        c           = randn(1)
                self.c            = theano.shared(c.astype('float32'))
                self.thetas_real  = theano.shared(thetas_real.astype('float32'))
                self.thetas_imag  = theano.shared(thetas_imag.astype('float32'))
                self.gammas_real  = theano.shared(gammas_real.astype('float32'))
                self.gammas_imag  = theano.shared(gammas_imag.astype('float32'))
	def get_filter_bank(self,T):
		return self.get_filters(T)#filters[::2],filters[1::2]
        def get_filters(self,T):
                """method to obtain one filter with length T"""
		ti              = th_linspace(0,1,self.S).dimshuffle([0,'x','x'])# THIS REPRESENTS THE MESH
		t               = th_linspace(0,1,T).dimshuffle(['x',0,'x'])#THIS REPRESENTS THE CONTINOUS TIME
		# APPLY CONSTRAINTS OF MEAN 0 AND BOUNDARY CONDITIONS
		thetas_real     = (self.thetas_real-self.thetas_real[:,1:-1].mean(1,keepdims=True))*self.mask.reshape((1,-1))
                thetas_imag     = (self.thetas_imag-self.thetas_imag[:,1:-1].mean(1,keepdims=True))*self.mask.reshape((1,-1))
		gammas_real     = self.gammas_real*self.mask.reshape((1,-1))#dimshuffle([0,'x'])
                gammas_imag     = self.gammas_imag*self.mask.reshape((1,-1))#dimshuffle([0,'x'])

		thetas_real = thetas_real.dimshuffle([1,'x',0])
                thetas_imag = thetas_imag.dimshuffle([1,'x',0])
                gammas_real = gammas_real.dimshuffle([1,'x',0])
                gammas_imag = gammas_imag.dimshuffle([1,'x',0])	

		#COMPUTE FILTERS BASED ON ACCUMULATION OF WINDOWED INTERPOLATION
		real_filter     = self.interp((t-ti[:-1])/(ti[1:]-ti[:-1]),thetas_real[:-1,:,:],gammas_real[:-1,:,:],thetas_real[1:,:,:],gammas_real[1:,:,:]).sum(0)
                imag_filter     = self.interp((t-ti[:-1])/(ti[1:]-ti[:-1]),thetas_imag[:-1,:,:],gammas_imag[:-1,:,:],thetas_imag[1:,:,:],gammas_imag[1:,:,:]).sum(0)

		real_filter/=real_filter.norm(2,axis=0,keepdims=True)
                imag_filter/=imag_filter.norm(2,axis=0,keepdims=True)
#		if(self.chirplet):
#			TT              = self.c[i].repeat(T)*float32(3.14159)*th_linspace(0,1,T)**2
#			real_filter     = real_filter*theano.tensor.cos(TT)-imag_filter*theano.tensor.sin(TT)
#                	imag_filter     = real_filter*theano.tensor.sin(TT)+imag_filter*theano.tensor.cos(TT)
                return real_filter.T,imag_filter.T
        def interp(self,t,pi,mi,pip,mip):
                values = ((2*t**3-3*t**2+1)*pi+(t**3-2*t**2+t)*mi+(-2*t**3+3*t**2)*pip+(t**3-t**2)*mip)
                mask   = theano.tensor.cast(theano.tensor.ge(t,0),'float32')*theano.tensor.cast(theano.tensor.lt(t,1),'float32')
                return values*mask
















def create_center_filter(row,filter_length,last_fb,max_length,class_function):
	deltaT = max_length-filter_length
        real_f,imag_f =class_function(filter_length)
        last_fb = theano.tensor.inc_subtensor(last_fb[2*row,deltaT/2:deltaT/2+filter_length],real_f)
        last_fb = theano.tensor.inc_subtensor(last_fb[2*row+1,deltaT/2:deltaT/2+filter_length],imag_f)
        return last_fb



def create_filter_banks(filter_class,T,J,Q):
        scales = array(2**(arange(J*Q+1,dtype='float32')/Q)).astype('float32')#all the scales
        Ts     = array([T*scale for scale in scales]).astype('int32')#all the filter size support
	Ts_t   = theano.shared(Ts)
	print "Lengths of the filters:",Ts
        filter_bank,_=theano.scan(fn = lambda row,filter_length,last_fb,max_length: create_center_filter(row,filter_length,last_fb,max_length,filter_class.get_filters),sequences=[theano.tensor.arange(J*Q+1,dtype='int32'),Ts_t],non_sequences=[Ts_t[-1]],n_steps=J*Q+1,outputs_info=theano.tensor.zeros(((J*Q+1)*2,Ts[-1]),dtype='float32'))
	filter_bank=filter_bank[-1]
        return filter_bank[::2],filter_bank[1::2],Ts[-1]


class SplineFilter1D(lasagne.layers.Layer):
        """class wrapping the lasagne toolbox for use of the spline filters"""
        def __init__(self,incoming,N,J,Q,S,type_='hermite',stride=1,pad='valid',nonlinearity=1,deterministic=0,renormalization=theano.tensor.max,initialization='gabor',chirplet=0,complex_=1, **kwargs):
		print 'SISI ON EST KEN'
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
                self.deterministic   = deterministic
                self.renormalization = renormalization
                self.initialization  = initialization
		self.chirplet        = chirplet
		if(family):
			if(complex_):
		                self.filter_class  = theano_hermite_complex(S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
                		real_filter_bank,imag_filter_bank,T= create_filter_banks(self.filter_class,N,J,Q)
		                self.filter_class  = filter_bank(J*Q+1,S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
		                T=S*2**J+1
	       		        print "T",T
		                real_filter_bank,imag_filter_bank = self.filter_class.get_filter_bank(T)
        		        self.real_filter_bank      = theano.tensor.cast(real_filter_bank.reshape((J*Q+1,1,T)),'float32')
             			self.imag_filter_bank      = theano.tensor.cast(imag_filter_bank.reshape((J*Q+1,1,T)),'float32')
             		        self.real_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q+1),filter_size=int(T),W=self.real_filter_bank,stride=int(stride),pad=int(pad),nonlinearity=None,b=None)
                		self.imag_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q+1),filter_size=int(T),W=self.imag_filter_bank,stride=int(stride),pad=int(pad),nonlinearity=None,b=None)

			else:
                                self.filter_class  = theano_hermite_complex(S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
                                real_filter_bank,imag_filter_bank,T= create_filter_banks(self.filter_class,N,J,Q)

                self.filter_class  = filter_bank(J*Q+1,S,deterministic=deterministic,renormalization=renormalization,initialization=initialization,chirplet=chirplet)
		T=S*2**J
		print "T",T
		real_filter_bank,imag_filter_bank = self.filter_class.get_filter_bank(T)
                self.real_filter_bank      = theano.tensor.cast(real_filter_bank.reshape((J*Q+1,1,T)),'float32')
                self.imag_filter_bank      = theano.tensor.cast(imag_filter_bank.reshape((J*Q+1,1,T)),'float32')
                self.real_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q+1),filter_size=int(T),W=self.real_filter_bank,stride=int(stride),pad=int(pad),nonlinearity=None,b=None)
                self.imag_layer            = lasagne.layers.Conv1DLayer(self.input_shape,num_filters=int(J*Q+1),filter_size=int(T),W=self.imag_filter_bank,stride=int(stride),pad=int(pad),nonlinearity=None,b=None)
        def get_output_for(self, input, **kwargs):
                real_output = self.real_layer.get_output_for(input)
                imag_output = self.imag_layer.get_output_for(input)
                return theano.tensor.sqrt(real_output**2+imag_output**2)
        def get_output_shape_for(self, input_shape):
                return self.real_layer.get_output_shape_for(input_shape)
        def get_params(self, **kwargs):
		if(self.deterministic): return []
                k= [self.filter_class.thetas_real,self.filter_class.thetas_imag,self.filter_class.gammas_real,self.filter_class.gammas_imag,self.filter_class.c]
		if(self.chirplet): return k
		else: return k[:-1]
        def get_filters(self):
                return [self.real_filter_bank,self.imag_filter_bank]




def mad(x):
	median_ = theano.tensor.sort(x,1)[:,x.shape[1]/2]
	return median_/0.6741

class ExtremDenoiseLayer(lasagne.layers.Layer):
	def __init__(self,incoming,sigma_eval = mad,**kwargs):
                super(ExtremDenoiseLayer, self).__init__(incoming, **kwargs)
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
		M    = theano.tensor.transpose(W)
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
		return output
	def get_output_shape_for(self,input_shape):
		return input_shape
	def get_params(self,**kwargs):
		return []


