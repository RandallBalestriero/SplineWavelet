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
	return test_auc.mean(0),test_auc.std(0),test_accu.mean(0),test_accu.std(0),train_error.mean(0),train_error.std(0)



def do_the_plot(x,y,std,c):
        plot(x,y,color=c,linewidth=3)
        fill_between(x,y-std,y+std,alpha=0.2,color=c)




def plot_it(path):
	pickles        = glob.glob(path)
	data_BULBUL    = load_files(select(pickles,['cagedbird_BULBUL','lr0.01']))
	data_CONV      = load_files(select(pickles,['cagedbird_CONV','lr0.01']))
	data_CONVGABOR = load_files(select(pickles,['cagedbird_GABOR','lr0.01']))

	data_splineBULBUL_random = load_files(select(pickles,['cagedbird_splineBULBUL','chirp0','lr0.01','random_log']))
        data_splineBULBUL_random_ap = load_files(select(pickles,['cagedbird_splineBULBUL','chirp0','lr0.01','random_apodized_log']))
	x500= arange(len(data_CONV[0]))
	x100 = arange(len(data_splineBULBUL_random[0]))
	figure()
	do_the_plot(x500,data_BULBUL[0],data_BULBUL[1],'g')
        do_the_plot(x500,data_CONV[0],data_CONV[1],'k')
        do_the_plot(x500,data_CONVGABOR[0],data_CONVGABOR[1],'b')
        do_the_plot(x500,data_splineBULBUL_random[0],data_splineBULBUL_random[1],'r')
        do_the_plot(x500,data_splineBULBUL_random_ap[0],data_splineBULBUL_random_ap[1],'m')
	suptitle('AUC')
        figure()
        do_the_plot(x500,data_BULBUL[2],data_BULBUL[3],'g')
        do_the_plot(x500,data_CONV[2],data_CONV[3],'k')
        do_the_plot(x500,data_CONVGABOR[2],data_CONVGABOR[3],'b')
        do_the_plot(x500,data_splineBULBUL_random[2],data_splineBULBUL_random[3],'r')
        do_the_plot(x500,data_splineBULBUL_random_ap[2],data_splineBULBUL_random_ap[3],'m')
        suptitle('ACUUU')
        show()



	show()	





plot_it('*.pkl')
