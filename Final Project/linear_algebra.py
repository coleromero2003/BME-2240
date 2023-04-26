import numpy as np

def main():
    #A = np.array([[4,-14,-12],[-14,-10,13],[-12,13,1]])
    #A = np.array([[7,8],[8,-10]])
    #A = np.array([[9,13,3,6],[13,11,7,6],[3,7,4,7],[6,6,7,10]])
    A = np.array([[5,4,3,2,1],[4,5,4,3,2],[3,4,5,4,3],[2,3,4,5,4],[1,2,3,4,5]])
    print(eigen(A)[0])
    print(eigen(A)[1])
    

def qrgs (A):
    #finding dimensions of the matrix
    M , N = A.shape 
    #Filling Q and R with zeroes
    R = np.zeros((N,N))
    u = np.zeros((M,N))
    e = np.zeros((M,N))
    #loop over each colum
    for k in range(N):
        if k == 0:
            u[:,k] = A[:,k]
            e[:,k] = (1/len_of_matrix(u[:,0]))*u[:,0]
        else:
            old = np.zeros((M,N))
            for i in range(k):
                old[:,0] += sum_2_mat_mult(e[:,i],(A[:,k]))*e[:,i]
            u[:,k] = A[:,k] - old[:,0]
            e[:,k] = (1/len_of_matrix(u[:,k]))*u[:,k]           
    R = makeR(A,e)
    return e, R

def linear_least_squares(A,B):
    # Get the QR decomposition of A
    Q, R = qrgs(A)
    #get qtb so we can solbe for x in next step
    QtB = np.asarray(np.dot(np.transpose(Q),np.transpose(B)))
    x = back_substitution(R,QtB)
    return x

def eigen(A, MAX_ITER=10000):
    #check if A is a square matrix
    M , N = A.shape 
    if M != N:
        print("A is not a square matrix")
        return None
    #create I vector for V so the first X will be correct
    V = I_maker(M)
    for i in range(MAX_ITER):
        Q,R = qrgs(A)
        A = np.dot(R,Q)
        V = np.dot(V,Q)
        if is_upper_triangular(A):
            break
    W = diagonal_ele(A)
    return W, V

def I_maker(M):
    I = np.zeros((M,M))
    for i in range(M):
        I[i][i] = 1
    return I

def diagonal_ele(A):
    M,N = A.shape
    W = np.zeros(M)
    for i in range(M):
        W[i] = A[i][i]
    return W

def back_substitution(R, b):
    # Get the number of rows and columns of R
    m, n = R.shape
    # Fill x with zeros 
    x = np.zeros(n)
    #used an equation I found online
    #also range can is (start,stop,step), made this really easy
    for i in (range(n-1,-1,-1)):
        sub = 0
        for j in range(i+1,m):
            sub += R[i][j] * x[j]
        x[i] = (b[i]-sub)/R[i][i]
    return x

def len_of_matrix(A):
    #take a matrix with a single row or column and returns the length of that line ||A||
    M = A.shape[0] 
    sum = 0
    for i in range(M):
        sum += A[i]**2
    return np.sqrt(sum)

def sum_2_mat_mult(A,B):
    #adds up all of the insides of a matrix that are being multiplied together
    M = A.shape[0] 
    sum = 0
    for i in range(M):
        sum += A[i] * B[i]
    return sum

def makeR(a,e):
    M , N = a.shape 
    R = np.zeros((N,N))
    for i in range(N):
        for j in range (N):
            if j  >= i:
                R[i,j] = sum_2_mat_mult(e[:,i],a[:,j])
    return R

def is_upper_triangular(A):
    #skip first row
    for i in range(1, A.shape[0]):
        for j in range(i):
            #checks every spot to see if small
            if abs(A[i][j]) > 10**(-30):
                # Return False
                return False
    # Return True
    return True



if __name__ == "__main__":
    main()