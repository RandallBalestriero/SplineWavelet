import pickle
from pylab import *
import glob

files = glob.glob('bird10saveNN0*.pkl')
print(files)
for f in files:
        n=open(f,'rb')
        a=pickle.load(n)
        n.close()
        print(f,max(a[4]),max(a[5]),max(a[6]),max(a[7]))
        if(len(a)>7):
                W1 = a[8]
                W11 = a[9]
                W2=a[10]
                W3=a[11]
                print(W1.shape)
                print(W1)
                for i in range(int(W1.shape[0]/16)):
                        subplot(int(W1.shape[0]/16),3,i*3+1)
                        plot(W1[i*16,0],linewidth=2)
                        if(i==0):
                                title('Hermite',fontsize=20)
                        xticks([])
                        subplot(int(W1.shape[0]/16),3,i*3+2)
                        plot(W2[i*16,0],linewidth=2)
                        if(i==0):
                                title('Lagrange',fontsize=20)
                        xticks([])
                        subplot(int(W1.shape[0]/16),3,i*3+3)
                        if(i==0):
                                title('CNN',fontsize=20)
                        plot(W3[i*16,0],linewidth=2)
                        xticks([])
                suptitle('Filter-Bank Comparison',fontsize=20)
        show()
#        subplot(121)
#        plot(a[0],'r')
#        plot(a[1],'b')
#        plot(a[2],'k')
#        subplot(122)
#        plot(a[3],'r')
#        plot(a[4],'b')
#        plot(a[5],'k')
#        title(f)
#        show()





