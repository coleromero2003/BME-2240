import linear_algebra as lin
import numpy as np
import matplotlib.pyplot as plt
import h5py
with h5py.File("Final Project\data.h5",) as hf:
    si = np.array(hf.get("si_images"))
with h5py.File("Final Project\data.h5",) as hf:
    b = np.array(hf.get("bvals"))
    g = np.array(hf.get("gradient"))
    s0 = np.array(hf.get("s0_image"))

def main():
    m,n,o = si.shape
    fa = np.zeros([m,n])
    map = np.zeros([m,n,3])
    for i in range(m):
            for j in range(n):
                fa[i,j] = FA(tensor(si,b,g,s0,i,j))
                map[i,j] = direction_map(tensor(si,b,g,s0,i,j))
    plt.imshow(fa,cmap = 'Greys', interpolation='nearest')
    plt.show()
    plt.imshow(map)
    plt.show()



def tensor(si,b,g,s0,m,n):
    y = -np.log((si[m,n,:]+1e-16))/(s0[m,n]+1e-16)/b
    Dx, Dy, Dz = lin.linear_least_squares(g.T,y)
    return np.array([[Dx*Dx,Dx*Dy,Dx*Dz],[0,Dy*Dy,Dy*Dz],[0,0,Dz*Dz]])

def FA(D):
    W, V = lin.eigen(D)
    idx = W.argsort()[::-1]
    W = W[idx]
    V = V[:,idx]
    fa = np.sqrt(((W[0]-W[2])**2 + (W[0]-W[2])**2 +(W[1]-W[2])**2)/(2*(W[0]**2 +W[1]**2 +W[2]**2 )))
    return fa

def direction_map(D):
    W, V = lin.eigen(D)
    idx = W.argsort()[::-1]
    W = W[idx]
    V = V[:,idx]
    fa = np.sqrt(((W[0]-W[2])**2 + (W[0]-W[2])**2 +(W[1]-W[2])**2)/(2*(W[0]**2 +W[1]**2 +W[2]**2 )))
    maximum = abs(np.array([V[0,2]*(1-fa),V[0,1]*(1-fa),V[0,0]*(1-fa)]))
    return maximum

if __name__ == "__main__":
    main()