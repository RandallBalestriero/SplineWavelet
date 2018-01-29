from unsup_detection_spline import *
from pylab import *
from scipy.io.wavfile import read
import csv
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
import cPickle


def trainit(X_train,n_epochs,MODEL,l_r):
        train_error    = []
        for n_epoch in xrange(n_epochs):
        	train_error.append(MODEL.train(X_train,float32(l_r)))
		print train_error[-1]
	return train_error

classes= sort(glob.glob('./bird/train_*'))
#classes =[c for c in classes if "sturnus_vulgaris" in c or "sylvia_atricapilla" in c or "troglodytes_troglodytes" in c or "turdus_merula" in c or "turdus_philomelos" in c or "turdus_viscivorus" in c]
DATA = []
names = []
for c in classes:
	print c
	data_files = glob.glob(c+'/0_*.wav')[0]
	print data_files
        Fs,x 	   = read(data_files)
	DATA.append((x[::2]+x[1::2]).astype('float32'))
	names.append(data_files.split("/")[-1])

DATA_TRAIN = []
DATA_TEST  = []
FILTERS_TRAIN = []
FILTERS_TEST  = []
for d in DATA:
	DATA_TRAIN.append(d[:len(d)/2])
	DATA_TEST.append(d[len(d)/2:])



N,J,Q,S    = 16,6,8,16
chirp      = 1
n_epochs   = 300
l_r_       = 0.1
log_       = 1
complex_   = 1
init_      ='random_apodized'



for i,j,k in zip(DATA_TRAIN,names,DATA_TEST):
	MODEL      = SCALO(x_shape=(1,len(i)),S=S,N=N,J=J,Q=Q,initialization=init_,renormalization=lambda x:x.norm(2),chirplet=chirp,complex_=complex_,log_=log_)
	error      = trainit(i.reshape((1,-1)),n_epochs,MODEL,l_r_)
	FILTERS_TRAIN.append(MODEL.get_filters())
        MODEL      = SCALO(x_shape=(1,len(k)),S=S,N=N,J=J,Q=Q,initialization=init_,renormalization=lambda x:x.norm(2),chirplet=chirp,complex_=complex_,log_=log_)
        error      = trainit(k.reshape((1,-1)),n_epochs,MODEL,l_r_)
        FILTERS_TEST.append(MODEL.get_filters())


f=open('data_bird_saved.pkl','wb')
cPickle.dump([FILTERS_TRAIN,FILTERS_TEST,DATA_TRAIN,DATA_TEST],f) 
f.close()  







