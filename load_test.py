import cPickle
from pylab import *
import sys 


name = sys.argv[-1]
print name
f = open(name,'rb')
F = cPickle.load(f)
print F
