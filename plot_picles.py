import glob
import cPickle
from pylab import *


def select(filelist,kwargs):
	if(len(kwargs)==0):
		return filelist
	return select([f for f in filelist if kwargs[0] in f],kwargs[1:])


def load_files(files):
	test_accu      = []
        test_auc       = []
	train_error    = []
	filters        = []
	representation = []
	for filename in files:
		f    = open(filename,'rb')
		data = cPickle.load(f)
		f.close()
		test_auc.append(data[1])
		train_error.append(data[0])
		test_accu.append(data[2])
		filters.append(data[3])
		representation.append(data[-1])
	test_auc = asarray(test_auc)
        test_accu = asarray(test_accu)
	train_error = asarray(train_error)
	return test_auc.mean(0),test_auc.std(0),test_accu.mean(0),test_accu.std(0),train_error.mean(0),train_error.std(0),filters,representation



def do_the_plot(x,y,std,c,label):
        plot(x,y,c,linewidth=4.5,label=label)
#        fill_between(x,y-std,y+std,alpha=0.2,color=c[-1])




def plot_it(path):
	pickles        = glob.glob(path)
	data_BULBUL    = load_files(select(pickles,['cagedbird_BULBUL','lr0.01']))
	data_CONV      = load_files(select(pickles,['NORMEDGE_CONV','lr0.01']))
	data_CONVGABOR = load_files(select(pickles,['NORMEDGE_GABOR','lr0.01']))
	data_splineBULBUL_random = load_files(select(pickles,['NORMEDGE_splineBULBUL','chirp0','lr0.01','random_log']))
        data_splineBULBUL_random_ap = load_files(select(pickles,['NORMEDGE_splineBULBUL','chirp0','lr0.01','random_apodized_log']))
        data_splineBULBUL_gabor = load_files(select(pickles,['NORMEDGE_splineBULBUL','chirp0','lr0.01','gabor']))
	x500= arange(len(data_CONV[0]))
	x100 = arange(len(data_splineBULBUL_random[0]))
	figure(figsize=(40,15))
	do_the_plot(x500,100*data_BULBUL[0],100*data_BULBUL[1],'k',label='Conv. MFSC')
        do_the_plot(x500,100*data_CONV[0],100*data_CONV[1],'b',label='Conv. init random')
        do_the_plot(x500,100*data_CONVGABOR[0],100*data_CONVGABOR[1],'--b',label='Conv. init Gabor')
        do_the_plot(x500,100*data_splineBULBUL_random[0],100*data_splineBULBUL_random[1],'r',label='Spline Conv. init random')
        #do_the_plot(x500,100*data_splineBULBUL_random_ap[0],100*data_splineBULBUL_random_ap[1],'-xr',label='Spline Conv. init random apodized')
        do_the_plot(x500,100*data_splineBULBUL_gabor[0],100*data_splineBULBUL_gabor[1],'--r',label='Spline Conv. init Gabor')
	xlabel('Epoch',fontsize=50)
	ylabel('$\%$',fontsize=50)
	xlim([0,len(x500)])
	xticks(fontsize=35)
	yticks(fontsize=35)
	#suptitle('Accuracy',fontsize=45)
	legend(loc=4,fontsize=40)
	tight_layout()
	savefig('auc.png')
	close()
#        figure()
#        do_the_plot(x500,data_BULBUL[2],data_BULBUL[3],'g')
#        do_the_plot(x500,data_CONV[2],data_CONV[3],'k')
#        do_the_plot(x500,data_CONVGABOR[2],data_CONVGABOR[3],'b')
        #do_the_plot(x500,data_splineBULBUL_random[2],data_splineBULBUL_random[3],'r')
#        do_the_plot(x500,data_splineBULBUL_random_ap[2],data_splineBULBUL_random_ap[3],'m')
#        suptitle('ACCUUUU')
	#figure()
	#for i in xrange(1):
	#	subplot(5,6,1+i*6)
	#	print shape(data_splineBULBUL_random_ap[-1][-1][-1])
	#	plot(data_splineBULBUL_random_ap[-1][-1][0][0][60+i][0],'b')
        #        plot(data_splineBULBUL_random_ap[-1][-1][0][1][60+i][0],'r')
        #        subplot(5,6,2+i*6)
        #        print shape(data_splineBULBUL_random_ap[-1][-1][-1])
        #        plot(data_splineBULBUL_random_ap[-1][-1][-1][0][60+i][0],'b')
        #        plot(data_splineBULBUL_random_ap[-1][-1][-1][1][60+i][0],'r')
        #        subplot(5,6,3+i*6)
        #        plot(data_CONV[-1][-1][0][60+i][0],'b')
        #        subplot(5,6,4+i*6)
        #        plot(data_CONV[-1][-1][-1][60+i][0],'b')
        #        subplot(5,6,5+i*6)
        #        plot(data_CONVGABOR[-1][-1][0][60+i][0],'b')
        #        subplot(5,6,6+i*6)
        #        plot(data_CONVGABOR[-1][-1][-1][60+i][0],'b')
		

	plot_filters(data_splineBULBUL_random_ap,1,'filtersplineapodized')
	plot_filters(data_splineBULBUL_random,1,'filtersplinerandom')
        plot_filters(data_splineBULBUL_gabor,1,'filtersplinegabor')
	plot_filters(data_CONV,1,'filterconv')
	plot_filters(data_CONVGABOR,1,'filterconvgabor')
	plot_repr(data_splineBULBUL_random_ap)
        plot_repr(data_splineBULBUL_random)
        plot_repr(data_splineBULBUL_gabor)
        plot_repr(data_CONV)
        plot_repr(data_CONVGABOR)
	plot_repr(data_BULBUL,1)




def plot_repr(X,opt=0):
	i=5
	if opt==0:
		fig = figure(figsize=(18,4))
		ax = fig.add_subplot(1,3,1)
		imshow(X[-1][0][-1][0][0][5][0],aspect='auto')
                xticks([])
                yticks([])
		fig.add_subplot(1,3,2)
	        imshow(X[-1][0][-1][0][0][8][0],aspect='auto')
                xticks([])
                yticks([])
		fig.add_subplot(1,3,3)
        	imshow(X[-1][0][-1][0][0][9][0],aspect='auto')
		xticks([])
		yticks([])
	else:
                fig = figure(figsize=(18,4))
                ax = fig.add_subplot(1,3,1)
                imshow(exp(X[-1][0][-1][0][0][5]),aspect='auto')
                xticks([])
                yticks([])
                fig.add_subplot(1,3,2)
                imshow(exp(X[-1][0][-1][0][0][8]),aspect='auto')
                xticks([])
                yticks([])
                fig.add_subplot(1,3,3)
                imshow(exp(X[-1][0][-1][0][0][9]),aspect='auto')
                xticks([])
                yticks([])



def plot_filters(X,complex_,name):
	i =5
	if complex_==1:
		fig = figure(figsize=(18,2.5))
	        ax= fig.add_subplot(1,3,1)
		plot(X[-2][0][0][0][70+i][0],'b',linewidth=3)
       	        plot(X[-2][0][0][1][70+i][0],'r',linewidth=3)
                xlim([0,len(X[-2][-1][0][0][70+i][0])])
		ax.set_yticklabels([])
	        ax.set_xticklabels([])
		title('Initialization',fontsize=25)
	
	        ax3= fig.add_subplot(1,3,2)
	        plot(X[-2][0][5][0][70+i][0],'b',linewidth=3)
                plot(X[-2][0][5][1][70+i][0],'r',linewidth=3)
                xlim([0,len(X[-2][-1][2][0][70+i][0])])
	        ax3.set_yticklabels([])
	        ax3.set_xticklabels([])
	        title('Epoch 5',fontsize=25)
	
	
		ax4 = fig.add_subplot(1,3,3)
	        plot(X[-2][0][-1][0][70+i][0],'b',linewidth=3)
                plot(X[-2][0][-1][1][70+i][0],'r',linewidth=3)
	        ax4.set_yticklabels([])
	       	ax4.set_xticklabels([])
                xlim([0,len(X[-2][-1][-1][0][70+i][0])])
		title('Last Epoch',fontsize=25)
		tight_layout()
		savefig(name+'.png')
		close()
	else:
                fig = figure()
                ax= fig.add_subplot(1,4,1)
                plot(X[-2][-1][0][60+i][0],c,linewidth=2)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                title('Initialization',fontsize=35)

                ax2= fig.add_subplot(1,4,2)
                plot(X[-2][-1][1][60+i][0],c,linewidth=2)
                ax2.set_yticklabels([])
                ax2.set_xticklabels([])
                title('Epoch 1',fontsize=35)

                ax3= fig.add_subplot(1,4,3)
                plot(X[-2][-1][2][60+i][0],c,linewidth=2)
                ax3.set_yticklabels([])
                ax3.set_xticklabels([])
                title('Epoch 2',fontsize=35)


                ax4 = fig.add_subplot(1,4,4)
                plot(X[-1][-1][-1][60+i][0],c,linewidth=2)
                ax4.set_yticklabels([])
                ax4.set_xticklabels([])
                title('Last Epoch',fontsize=35)




plot_it('*.pkl')
show()
