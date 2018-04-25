from bird_detection_spline import *
from pylab import *
from scipy.io.wavfile import read
import csv
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import cPickle
import base


def trainit(X_train,Y_train,X_test,Y_test,batch_size,n_epochs,MODEL,index_0,index_1,l_r):
        n_batch        = X_train.shape[0]/batch_size
        train_error    = []
        test_error     = []
        filters        = []
        representation = []
	accuracy           = []
        for n_epoch in xrange(n_epochs):
                if(n_epoch<10 or n_epoch==(n_epochs-1)):
                        filters.append(MODEL.get_filters())
                for batch in xrange(n_batch):
                        index_0_batch = random.sample(index_0,batch_size/2)
                        index_1_batch = random.sample(index_1,batch_size/2)
                        index_batch   = concatenate([index_0_batch,index_1_batch])
                        x_batch = X_train[index_batch]
                        y_batch = Y_train[index_batch]
                        if(isinstance(MODEL,BULBUL)):
                                x_batch = asarray([base.logfbank(x,samplerate=22050,winlen=0.046,winstep=0.010,nfilt=80,nfft=1024,lowfreq=10,highfreq=11000).T for x in x_batch]).astype('float32')
                        train_error.append(MODEL.train(x_batch,y_batch,l_r.astype('float32')))
                        if batch%40 == 0:
                                print 'batch n_',batch,'out of',n_batch,': ','training error=', train_error[-1]
		auc,accu = testit(X_test,Y_test,batch_size,MODEL)
                test_error.append(auc)
		accuracy.append(accu)
                print "epoch n_",n_epoch,' AUC= ', test_error[-1]
	index_0 = find(Y_test==0)
        index_1 = find(Y_test==1)
	for i in xrange(5):
                index_batch   = concatenate([index_0[i*batch_size/2:(i+1)*batch_size/2],index_1[i*batch_size/2:(i+1)*batch_size/2]])
                x_batch = X_train[index_batch]
                y_batch = Y_train[index_batch]
		if(isinstance(MODEL,BULBUL)):
                	x_batch = asarray([base.logfbank(x,samplerate=22050,winlen=0.046,winstep=0.010,nfilt=80,nfft=1024,lowfreq=10,highfreq=11000).T for x in x_batch]).astype('float32')
		representation.append([MODEL.get_repr(x_batch),y_batch])
        f = open(name,'wb')
        cPickle.dump([train_error,test_error,accuracy,filters,representation],f)
        f.close()

def testit(X,Y,batch_size,MODEL):
        n_batch = X.shape[0]/batch_size
        y_hat   = []
	accu    = []
        for batch in xrange(n_batch):
                if(isinstance(MODEL,BULBUL)):
                        x_batch = asarray([base.logfbank(x,samplerate=22050,winlen=0.046,winstep=0.010,nfilt=80,nfft=1024,lowfreq=10,highfreq=11000).T for x in X[batch*batch_size:(batch+1)*batch_size].astype('float32')]).astype('float32')
                else:
                        x_batch = X[batch*batch_size:(batch+1)*batch_size].astype('float32')
		y_batch = Y[batch*batch_size:(batch+1)*batch_size].astype('int32')
                accu.append(MODEL.test(x_batch,y_batch))
                y_hat.append(MODEL.predict(x_batch))
	accuracy = array(accu).mean()
	print 'ACCURACY ',accuracy
        y_hat = concatenate(y_hat,axis=0)
        y_hat = vstack([y_hat.reshape((1,-1)),1-y_hat.reshape((1,-1))]).T
        return roc_auc_score(Y[:shape(y_hat)[0]],y_hat.argmin(axis=1)),accuracy


## MAIN ##


csv_file    = open('ff1010bird_metadata.csv','rb')
label_csv   = csv.reader(csv_file)
data_list   = vstack(list(label_csv))
names_wav   = data_list[1:,0]
labels      = data_list[1:,1]
complex_    = 1# '0 1'
init_       = sys.argv[-1] # 'random' 'apodized_random' 'gabor'
chirp       = int(sys.argv[-2])# '0 1'
MODEL_      = sys.argv[-3] # 'BULBUL CONV SPLINE GABOR'
l_r_        = float32(sys.argv[-4])
log_        = int(sys.argv[-5])
batch_size_ = 10
n_data_     = int(sys.argv[-6])
names_wav   = names_wav[:7000]
labels      = labels[:7000]
DATA = []
aug_        = 0


for i in xrange(7000):
	data_files = sort(glob.glob('./wav/'+names_wav[i]+'.wav'))
        Fs,x 	   = read(data_files)
	x          = x[0:len(x)-len(x)%2]
	DATA.append(x[0::2]+x[1::2])
N,J,Q,S    = 16,5,16,16
n_epochs   = 150

seed(3)
p = permutation(7000,n_data_)

for i in xrange(5):
	X_train,X_test,Y_train,Y_test 	= train_test_split(vstack(DATA)[p],labels[p],test_size=0.33,stratify=labels[p],random_state=10+i)
	X_train 		      	= X_train.astype('float32')
	X_test 			      	= X_test.astype('float32')
	Y_train 		      	= Y_train.astype('int32')
	Y_test  			= Y_test.astype('int32')
	X_train 		       -= X_train.mean(axis=1).reshape((-1,1))
	X_train 		       /= X_train.max(axis=1).reshape((-1,1))
	X_test 			       -= X_test.mean(axis=1).reshape((-1,1))
	X_test 	   		       /= X_test.max(axis=1).reshape((-1,1))
	if(MODEL_ == 'BULBUL'):
		MODEL      = BULBUL(x_shape=(batch_size_,80,995),S=S,N=N,J=J,Q=Q,aug=aug_)
	elif(MODEL_ == 'CONV'):
	        MODEL      = conv_BULBUL(x_shape=(batch_size_,shape(X_train)[1]),S=S,N=N,J=J,Q=Q,aug=aug_,log_=log_)
	elif(MODEL_ == 'splineBULBUL'):
	        MODEL      = spline_BULBUL(x_shape=(batch_size_,shape(X_train)[1]),S=S,N=N,J=J,Q=Q,deterministic=0,initialization=init_,renormalization=lambda x:x.norm(2),chirplet=chirp,aug=aug_,complex_=complex_,log_=log_)
        elif(MODEL_ == 'GABOR'):
                MODEL      = GABOR_BULBUL(x_shape=(batch_size_,shape(X_train)[1]),S=S,N=N,J=J,Q=Q,aug=aug_,log_=log_)
	name = 'NORMEDGE_'+MODEL_+str(i)+'_lr'+str(l_r_)+'_chirp'+str(chirp)+'_init'+init_+'_log'+str(log_)+'.pkl'
	print name
	train_size =  shape(X_train)[0]
	Y_train = array(Y_train)
	index_0 = find(Y_train==0)
	index_1 = find(Y_train==1)
	trainit(X_train,Y_train,X_test,Y_test,batch_size_,n_epochs,MODEL,index_0,index_1,l_r_)







