import glob
import cPickle
from pylab import *


def select(filelist,kwargs):
	if(len(kwargs)==0):
		return filelist
	return select([f for f in filelist if kwargs[0] in f],kwargs[1:])


def load_files(files):
	test_accu      = []
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
	filters = asarray(filters)
	return test_auc.mean(0),test_auc.std(0),test_accu.mean(0),test_accu.std(0),train_error.mean(0),train_error.std(0)



def plot_it(path):
	pickles        = glob.glob(path)
	data_BULBUL    = load_files(select(pickles,['_BULBUL','lr0.001']))
	data_CONV      = load_files(select(pickles,['_CONV','log1','lr0.01']))
	data_splineBULBUL_chirp0 = load_files(select(pickles,['_splineBULBUL','chirp0','lr0.005','random_log']))
        data_splineBULBUL_chirp1 = load_files(select(pickles,['_splineBULBUL','chirp1','lr0.005','random_log']))
	x500= arange(len(data_CONV_log0[0]))
	x100 = arange(len(data_splineBULBUL_chirp0[0]))
	figure()
	plot(x500,data_BULBUL[0],color='b')
	fill_between(x500,data_BULBUL[0]-data_BULBUL[1],data_BULBUL[0]+data_BULBUL[1],alpha=0.5,color='b')
	plot(x500,data_CONV_log0[0],color='g')
	fill_between(x500,data_CONV_chirp0[0]-data_CONV_log0[1],data_CONV_log0[0]+data_CONV_log0[1],color='g')
        plot(x100,data_splineBULBUL_log0[0],color='o')
        fill_between(x100,data_splineBULBUL_log0[0]-data_splineBULBUL_log0[1],data_splineBULBUL_log0[0]+data_splineBULBUL_log0[1],color='o')
        plot(x100,data_splineBULBUL_log1[0],color='r')
        fill_between(x100,data_splineBULBUL_log1[0]-data_splineBULBUL_log1[1],data_splineBULBUL_log1[0]+data_splineBULBUL_log1[1],color='r')
	suptitle('AUC')
	figure()
        plot(x500,data_BULBUL[2],color='b')
        fill_between(x500,data_BULBUL[2]-data_BULBUL[3],data_BULBUL[2]+data_BULBUL[3],alpha=0.5,color='b')
        plot(x500,data_CONV_log0[2],color='g')
        fill_between(x500,data_CONV_chirp0[2]-data_CONV_log0[3],data_CONV_log0[2]+data_CONV_log0[3],color='g')
        plot(x100,data_splineBULBUL_log0[2],color='o')
        fill_between(x100,data_splineBULBUL_log0[2]-data_splineBULBUL_log0[3],data_splineBULBUL_log0[2]+data_splineBULBUL_log0[3],color='o')
        plot(x100,data_splineBULBUL_log1[2],color='r')
        fill_between(x100,data_splineBULBUL_log1[2]-data_splineBULBUL_log1[3],data_splineBULBUL_log1[2]+data_splineBULBUL_log1[3],color='r')
        suptitle('AUC')
	show()	


plot_it('*.pkl')
