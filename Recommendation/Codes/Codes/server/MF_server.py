import nimfa
import math
import random
import numpy as np

class MF_Recommendation():
    
    def read_form_file_and_generate_main_data_matrix(self, file_address):
        movie_lens_file = open(file_address,'r');
        count = 0;
        user_id_list = [];
        movie_id_list = [];
        for line in movie_lens_file:
            '''if(count > 5 ):
                break;
            else:
                count += 1;'''
            line = line.split("\t");
            #print int(line[0]), "\t", int(line[1])
            user_id_list.append( int(line[0]) );
            movie_id_list.append( int(line[1]) );
            
        #print "Total number of ratings: ", len(user_id_list)
        #print "Number of unique users: ", max(user_id_list);
        #print "Number of unique movies: ", max(movie_id_list), "\n";
        movie_lens_file.close();
        
        #intialize the data matrix
        main_data = np.zeros(shape=( max(user_id_list) , max(movie_id_list) ))  #every row will be an user, every column will be a movie
        file1 = open(file_address,'r');
        for line in file1:
            line = line.split("\t");
            row_num = int(line[0]);
            col_num = int(line[1]);
            main_data[row_num-1][col_num-1] = float(line[2])/float(5);
        
        np.savetxt("main_data_mf.csv", main_data, delimiter=",");
        file1.close();
        return main_data;
    
    def get_data_sparsity(self, data):
        one_count = 0
        for row in data:
            for element in row:
                if (element == 1 ):
                    one_count += 1
        
        sparsity = one_count / ( float( len(data)*len(data[0]) ) )
        print "Data Density :", sparsity*100, "%";
        
    def make_train_and_test_sample(self, data, percentage):
        num_row = len(data)
        num_col = len(data[0])
        
        one_count = 0;
        replace_count = 0;
        test_indices_file = open("test_indices_mf.csv", "wb");
        for i in range(0,num_row):
            for j in range(0, num_col):
                if ( int(data[i][j]) > 0 ):
                    one_count += 1;
                    if (random.randint(1,100) < percentage):
                        data[i][j] = 0;
                        indices = "%d,%d\n"%(i,j);
                        test_indices_file.write(indices);
                        replace_count += 1;
                    #else:
                    #    data[i][j] = 0;
        
        #print "Replace count: ", replace_count, "(", 100*replace_count/float(one_count), "% of total rating)";
        np.savetxt("train_data_mf.csv", data, delimiter=",");
        return data
    
    def calculate_rmse(self, main_data, factorized_data, test_indices):
        sum_difference = 0.0;
        for line in test_indices:
            row = int(line[0]);
            col = int(line[1]);
            diff = float(main_data[row, col]) - float(factorized_data[row, col]);
            diff = math.pow(diff, 2);
            sum_difference += diff
        
        mse = sum_difference / float(len(test_indices))
        rmse = math.sqrt(mse);
        return rmse;
    
    def calculate_rmse_1(self, main_data, factorized_data, test_indices):
        sum_difference = 0.0;
        for line in test_indices:
            row = int(line[0]);
            col = int(line[1]);
            diff = float(main_data[row, col]) - float(factorized_data[row, col]);
            diff = math.pow(diff, 2);
            sum_difference += diff
        mse = sum_difference / float(len(test_indices))
        rmse = math.sqrt(mse);
        return rmse;
        

if __name__ == '__main__':
    print "Hello world!\n"
    obj_mf = MF_Recommendation();
    file_address = '../../Data/ml-100k/u1.base'
    #file_address = 'u1.base'
    main_data = obj_mf.read_form_file_and_generate_main_data_matrix(file_address);
    train_data = obj_mf.make_train_and_test_sample(main_data,20);
    test_indices = np.genfromtxt('test_indices_mf.csv', dtype= float, delimiter=',');
    N = len(train_data)
    M = len(train_data[0])
    K = 9;
    iteration_number=100; alpha=0.1; beta=0.02;

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    R = np.array(train_data);
    
    P = np.array(P, np.float);
    Q = np.array(Q, np.float);
    R = np.array(R, np.float);
    
    Q = Q.T
    for iteration in range(0, iteration_number):
        for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                if (R[i][j] > 0 ) :
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        temp1_p_beta = beta * P[i][k];
                        temp1_p = float(0)
                        temp1_p = float(2 * eij * Q[k][j] - temp1_p_beta);
                        P[i][k] = float(P[i][k] + alpha * temp1_p)
                        #print P[i][k]
                        temp1_q_beta = beta * Q[k][j]
                        temp1_q = float(0)
                        temp1_q = float(2 * eij * P[i][k] - temp1_q_beta)
                        Q[k][j] = float(Q[k][j] + alpha * temp1_q)
                    
        eR = np.dot(P,Q)
        #e = 0
        #for i in xrange(len(R)):
        #        for j in xrange(len(R[i])):
        #            if R[i][j] > 0:
        #                e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
        #                for k in xrange(K):
        #                    e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        #if e < 0.01:
        #    break
        print "Iteration: ", iteration
        
    factorized_data = np.dot(P, Q);
    
    main_data = np.genfromtxt('main_data_mf.csv', dtype= float, delimiter=',');
    rmse = obj_mf.calculate_rmse_1(main_data, factorized_data, test_indices)
    print "k:", K, "\tRMSE:", rmse;
    
    