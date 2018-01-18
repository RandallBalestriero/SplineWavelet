from pylab import *
from utils import *
from models import *
#exec(open('utils.py').read())
#exec(open('models.py').read())






from scipy.io.wavfile import read




x=read('170202_03_Dcall_Blue_Bombyx.wav')
x=x[1][::2**8,0]
x=x.astype('float32')
x-=x.mean()
x/=x.max()

N,J,Q,S = 16,3,16,16


def learn_scalo(x,chirplet):
	cnn1    = scalo(x_shape=(1,len(x)),S=S,N=N,J=J,Q=Q,type_='hermite',deterministic=0,initialization='random_apodized',renormalization=lambda x:x.norm(2),chirplet=chirplet)
	REAL_INIT=cnn1.get_filters()[0][-1,0,:]
	IMAG_INIT=cnn1.get_filters()[1][-1,0,:]
	loss = []
	for n in range(2000):
		loss.append(cnn1.train(x.astype('float32').reshape((1,-1))))
	 	print loss[-1]
	REAL_FINAL=cnn1.get_filters()[0][-1,0,:]
	IMAG_FINAL=cnn1.get_filters()[1][-1,0,:]
	return REAL_INIT+1j*IMAG_INIT,REAL_FINAL+1j*IMAG_FINAL,cnn1.get_scalo(x.reshape((1,-1)))[0],loss


loss_00=[]
loss_11=[]

figure()

for i in xrange(5):
	init,final,scalo_,loss_0 = learn_scalo(x,0)
	loss_00.append(loss_0)
	subplot(4,5,i+1)
	plot(real(init))
	plot(imag(init))
	xticks([])
	yticks([])
        if(i==2):
                title('Before and After Learning WITHOUT Chirpness',fontsize=25)
	subplot(4,5,i+6)
        plot(real(final))
        plot(imag(final))
	xticks([])
	yticks([])
        init,final,scalo_,loss_1 = learn_scalo(x,1)
	loss_11.append(loss_1)
        subplot(4,5,i+11)
        plot(real(init))
        plot(imag(init))
        xticks([])
        yticks([])
        if(i==2):
                title('Before and After Learning WITH Chirpness',fontsize=25)
        subplot(4,5,i+16)
        plot(real(final))
        plot(imag(final))
        xticks([])
        yticks([])


figure()
for i in xrange(5):
	plot(loss_00[i],'b')
	plot(loss_11[i],'r')
show()






