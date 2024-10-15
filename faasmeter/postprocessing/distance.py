import numpy as np

""" Computing the distance/error/accuracy using various measures """ 

def cosin(T, V):
    """ Cosine similarity between target and observation vector V """
    angle = np.dot(T, V)
    normaliz  = np.linalg.norm(T) * np.linalg.norm(V)
    return angle/normaliz

def L2normN(T, V, N):
    # N is the num invocations
    # Compute the energy shares first
    Tn = np.dot(T, N)
    Tnn = Tn/np.sum(Tn)
    
    Vn = np.dot(V, N)
    Vnn = Vn/np.sum(Vn)

    return np.linalg.norm(Tnn - Vnn) 
    
def L2norm(T, V):
    dn = np.linalg.norm(T - V)
    return dn / np.linalg.norm( V ) 

def indiv_err(T, V):
    diff = T - V
    return diff/V 

def varJT(J, T, N):
    """ J: sequence of energy footprints of functions, over time. 
    T: latencies 
    N: Number of invocations of each function """ 
    # Standard error(x) = var(x)/sqrt(N) 
    # What is N is tricky. Either number of timesteps 
    # varJT = StErr(J)/StErr(T)
    # return mean, var of varJT 
    pass 

if __name__ == "__main__":
    a = np.array( [3,2,4,1,5] )
    b = np.array( [3,4,4,9,5] )
    n = np.array( [10,20,10,11,12] )

    print( a, b, n )

    c = cosin( a, b )
    print( c )

    c = L2normN( a, b, n )
    print( c )

    c = L2norm( a, b )
    print( c )

##################################################
