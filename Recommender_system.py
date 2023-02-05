import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings("ignore")

def CUR(matrix,k):
    '''
        matrix : (m x n )
        k : dimension to reduce
        Output: C(m x k) , U(k x k) , R(k x n)
        
    '''
    
    m = matrix.shape[0]
    n = matrix.shape[1]
    
    if k >m or k>n:
        print("No possible CUR decompostion as k is greater than m,n")
        return None,None,None
    def pseudoInverse(A):
        U,sigma,VT = np.linalg.svd(A,full_matrices=False)
        igma = np.array([ 1/item if item >0.1 else 0 for item in sigma])
        V = VT.T
        UT = U.T
        pseudoA = np.matmul(np.matmul(V,np.diag(igma)),UT)
        return pseudoA

    def columnSelect(A,k):
        m = A.shape[0]
        n = A.shape[1]
        A_sq_el =  A**2
        d = np.sum(A_sq_el)
        l2_norm = np.sum(A_sq_el,axis=0)   # l2 norm  square for each column... 
        prob_col = l2_norm/d

        d = {}
        for i in range(n):
            d[prob_col[i]]=i
            
        cols= []
        columns=[]
        B = A.T
        for item in sorted(d.keys(),reverse=True):
            if k:
                k-=1
                cols.append(B[d[item]])
                columns.append(d[item])
        return np.array(cols).T,columns
    
    W = np.zeros(shape=(k,k))
    C,cols = columnSelect(matrix,k)
    R,rows = columnSelect(matrix.T,k)
    for i in range(len(rows)):
        for j in range(len(cols)):
            W[i][j] = matrix[rows[i]][cols[j]]
    R=R.T

    
    U= pseudoInverse(W)
    
    
    return C, U,R


def matrix_factorization(matrix, P, Q, K, mx_itr=1000, alpha=0.003, lmda=0.05):
    '''
    matrix : rating matrix
    P: n_users * K 
    Q: n_movies * K 
    K: latent features
    mx_itr: iterations
    alpha: learning rate
    lmda: regularization parameter
    '''
    
    epoch =0
    while(epoch < mx_itr):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] > 0:
                    #error for ith user and jth movie latent vector
                    error_ij = matrix[i][j] - np.dot(P[i,:],Q[j,:])

                    for k in range(K):
                        #gradient update    
                        P[i][k] = P[i][k] + alpha * (2 * error_ij * Q[j][k] - lmda * P[i][k])
                        Q[j][k] = Q[j][k] + alpha * (2 * error_ij * P[i][k] - lmda * Q[j][k])
                    

        
        mt = np.dot(P,Q.T)
        loss = np.linalg.norm(matrix-mt)
        
        epoch +=1
        print("Total loss is {} after {} iterations ".format(loss,epoch) ,end='\r')
        if loss<1:
            return
        
    return P, Q




def get_matrix(X,y):
    users = np.array(X.userId.values)
    movies=np.array(X.movieId.values)
    ratings = np.array(y)
    movieToIndex = sorted(set(movies))
    d= {}
    for i in range(len(movieToIndex)):
        d[movieToIndex[i]] =i
    matrix = np.zeros((users.max()+1,len(movieToIndex)))
    for i in range(X.shape[0]):
        matrix[users[i]][d[movies[i]]] = ratings[i]
    return matrix,d


def run_dataset(dataset_path,dataset="small"):
    df = pd.read_csv(dataset_path)
    df.head()
    X= df[['userId','movieId']]
    y= df['rating']
    times =[]
    matrix,_ = get_matrix(X,y)

    #svd
    print("\n--------------------------------  SVD decomposition ------------------------------\n")

    if(dataset=="small"):
        st_time = time.time()
        u, sigma, vt = np.linalg.svd(matrix)
        print("Top 20 singular values are \n{}".format(sigma[:20]))
        sigma_sq = sigma**2
        sigma_sq_sum = sum(sigma_sq)
        svd_loss =[]
        for item in sigma_sq:
            sigma_sq_sum = sigma_sq_sum - item
            if sigma_sq_sum<0: break
            svd_loss.append(np.sqrt(sigma_sq_sum))
        print("\nLoss of data in matrix with k latent factors Graph Generated")

        fig = plt.figure(figsize=(15,10))
        plt.plot(range(len(svd_loss)),svd_loss)
        plt.xlabel("k (latent factors)")
        plt.ylabel("loss of data in matrix with k latent factors")
        plt.savefig("svd_loss")
        
        nd_time = time.time()
        #time
        times.append(nd_time-st_time)
        print("Time taken by svd  is {} seconds  :".format(times[0]))

        

    print("\n--------------------------------  CUR decomposition ------------------------------\n")

    #CUR
    if(dataset=='small'):
        st_time = time.time()
        cur_loss= []
        for i in range(1,501):
            C, U, R = CUR(matrix,i)
            B = np.matmul(C,np.matmul(U,R))
            cur_loss.append(np.linalg.norm(matrix-B))
            print("CUR loss when k={} generated".format(i),end='\r')
            
        #cur loss plot from main matrix
        plt.plot(range(1,501),cur_loss)
        fig = plt.figure(figsize=(15,10))
        plt.plot(range(1,501),cur_loss)
        plt.xlabel("k (latent factors)")
        plt.ylabel("loss of data in matrix with k latent factors")
        plt.savefig("cur_loss")
        nd_time = time.time()
        times.append(nd_time-st_time)
        print("Time taken by CUR  is {} seconds  :".format(times[1]))

    print("\n--------------------------------  PQ decomposition ------------------------------\n")
    if(dataset=='small'):
        
        st_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        matrix_t,movie_map = get_matrix(X_train,y_train)
        k=30
        P = np.random.rand(matrix_t.shape[0],k)
        Q = np.random.rand(matrix_t.shape[1],k)

        P,Q =matrix_factorization(matrix_t, P, Q, k, mx_itr=500, alpha=0.03, lmda=0.5)

        mt=  np.matmul(P,Q.T)
        total_error= np.linalg.norm(matrix_t-mt)
        print()
        print("Total loss  for PQ decomposition of matrix for 30 latent vectors: {}".format(total_error))

        train_err = 0
        test_err=0

        for item in np.array(X_train):
            train_err = train_err + (matrix_t[item[0]][movie_map[item[1]]] - mt[item[0]][movie_map[item[1]]])**2

        n= X_train.shape[0]
        print("\n------------- Training Loss ----------------\n")
        print("Total_MSE_loss :{} with {} enteries".format(train_err,n))        
        print("MSE_loss for each Entry :{}".format(train_err/X_train.shape[0]))  

        n = X_test.shape[0]
        X_test_arr = np.array(X_test)
        y_test_arr = np.array(y_test)
        p=0
        for i in range(n):
            item = X_test_arr[i]
            if item[1] in movie_map.keys(): 
                test_err = test_err + (y_test_arr[i] - mt[item[0]][movie_map[item[1]]])**2
            else: p= p+1

        print("\n------------- Test Data Loss ----------------\n")
        print("Total_MSE_loss :{} with {} enteries".format(test_err,n-p))        
        print("MSE_loss for each Entry :{} \n".format(test_err/(n-p)))  
        
        nd_time = time.time()
        times.append(nd_time-st_time)
        print("Time taken by PQ decompostion over 500 iterations for 30 latent feautres is {} seconds  :\n".format(times[2]))

    







if __name__=='__main__':
    run_dataset(dataset_path='../data/ml-latest-small/ratings.csv')

    

