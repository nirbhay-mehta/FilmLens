import numpy

class test():
     
    def matrix_factorization(R, N, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
        Q = Q.T
        for step in range(0, steps):
            for i in range(0, N):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = numpy.dot(P,Q)
            e = 0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in xrange(K):
                            e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
            if e < 0.001:
                break
        return P, Q.T
    
if __name__ == '__main__':
    obj = test();
    R = [[5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],]
    R = numpy.array(R); print R;
    N = len(R)
    
    M = len(R[0])
    K = 2;
    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)
    #nP, nQ = obj.matrix_factorization(R, N, P, Q, K)
    #nR = numpy.dot(nP, nQ.T)
    
    iteration_number=500; alpha=0.02; beta=0.02;
    Q = Q.T
    for iteration in range(0, iteration_number):
        for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                if (R[i][j] > 0 ) :
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
                    
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in xrange(K):
                            e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.01:
            break
        
    
    #matrix factorization is complete
    R1 = numpy.dot(P, Q)
    print "\n\n", R1
    
    
    
    
    
    