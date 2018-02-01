from pylab import *
import cPickle



if(0):
	f=open('data_bird_saved.pkl','rb')
	filters_train,filters_test,data_train,data_test = cPickle.load(f)
	f.close()

	figure()
	for i in xrange(10):
		subplot(1,10,i+1)
		print shape(filters_train)
		plot(filters_train[i][0][-1][0],'b')
	        plot(filters_train[i][1][-1][0],'r')
	        xticks([])
	        yticks([])
	        title(str(i),fontsize=20)
	show()

	filters_train = asarray(filters_train)
	filters_test  = asarray(filters_test)

	def entropy(x):
		return abs(x).max()

	figure()
	corr_matrix = zeros((len(data_train),len(data_train)))
	for i in xrange(len(data_train)):
		for j in xrange(len(data_train)):
			corr_matrix[i,j]=entropy(ifft(fft(data_train[j]).reshape((1,-1))*fft(filters_train[i,0,:,0]+1j*filters_train[i,1,:,0],n=len(data_train[j]))))
		print corr_matrix[i].argmin()
	
	imshow(corr_matrix)
	show()


if(1):
        f=open('octave_data_bird_saved.pkl','rb')
        filters_train,filters_test,data_train,data_test = cPickle.load(f)
        f.close()

        figure()
	for j in xrange(4):
        	for i in xrange(len(filters_train[0])):
                	subplot(len(filters_train[0]),4,i*4+1+j)
                	plot(filters_train[j][i][0][-1][0],'b')
                	plot(filters_train[j][i][1][-1][0],'r')
                	xticks([])
                	yticks([])
			if(i==0):
	                	title("File "+str(j),fontsize=20)
        show()






























