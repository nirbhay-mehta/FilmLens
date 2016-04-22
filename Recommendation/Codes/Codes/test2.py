import nimfa
import numpy as np

V = nimfa.examples.medulloblastoma.read(normalize=True)
V = [
    [5,3,5,5],
    [5,5,0,5],
    [1,1,5,5],
    [0,0,5,0],
    [0,1,5,4],
    ]
V = np.array(V)


#fctr = nimfa.mf(V, seed='random_vcol', method='snmf', rank=3, max_iter=2)
#fctr_res = nimfa.mf_run(fctr)
#W = fctr_res.basis()
#H = fctr_res.coef()
##print H
#print V, "\n\n"
#
#print np.dot(W,H)


#print 'Rss: %5.4f' % fctr_res.fit.rss()

x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
x = np.array(x, np.complex)

print x.shape
P = np.random.rand(3,3)
print P