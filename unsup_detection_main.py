from bird_detection_spline import *
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
classes =[c for c in classes if "sturnus_vulgaris" in c or "sylvia_atricapilla" in c or "troglodytes_troglodytes" in c or "turdus_merula" in c or "turdus_philomelos" in c or "turdus_viscivorus" in c]
DATA = []
names = []
for c in classes:
	print c
	data_files = glob.glob(c+'/0_*.wav')[0]
	print data_files
        Fs,x 	   = read(data_files)
	DATA.append((x[::2]+x[1::2]).astype('float32'))
	DATA[-1]-=DATA[-1].mean()
	DATA[-1]/=DATA[-1].max()
	names.append(data_files.split("/")[-1])

print DATA
print names

N,J,Q,S    = 16,6,8,8
chirp   = 1
n_epochs   = 300
l_r_ = 0.1
log_ = 1
complex_ = 1
init_='gabor'

for i,j in zip(DATA,names):
	MODEL      = SCALO(x_shape=(1,len(i)),S=S,N=N,J=J,Q=Q,initialization=init_,renormalization=lambda x:x.norm(2),chirplet=chirp,complex_=complex_,log_=log_)
	error      = trainit(i.reshape((1,-1)),n_epochs,MODEL,l_r_)
	filtersr,filtersi    = MODEL.get_filters()#[:,:,0,:]
	repren     = MODEL.get_repr(i.reshape((1,-1)))[0]
	print repren.shape
#	subplot(211)
#	imshow(repren,aspect='auto')
#	subplot(212)
#	imshow(filtersr[:,0,:],aspect='auto')
#	show()
	savetxt('logscalo_'+j[:-3]+'csv',repren,delimiter=',')
        savetxt('realfilters_'+j[:-3]+'csv',filtersr[:,0,:],delimiter=',')
        savetxt('imagfilters_'+j[:-3]+'csv',filtersi[:,0,:],delimiter=',')






