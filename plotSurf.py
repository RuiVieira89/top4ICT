
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms

from scipy.optimize import minimize

from function import function
from helpFun import funVar

def plotSurf(A, minimum):

	[Nx, Ny, ne, E, h, a, b] = funVar()

	x = np.arange(0,a+a/Nx,a/Nx)
	y = np.arange(0,b+b/Ny,b/Ny)
	X,Y = np.meshgrid(x, y)
	Z = function([X, Y])

	plt.ion()
	fig = plt.figure(1, figsize=(8*a/b,6), clear=True)
	ax = fig.add_subplot(1, 1, 1)

	im = ax.contourf(X, Y, Z, levels=100)

	if A != []:
		ax.scatter(A[:,0], A[:,1], marker='x', c='r')

		for i in range(0,A.shape[0]):
			s = f"X{i}"
			ax.text(A[i,0], A[i,1], s, fontsize=12)

	if minimum != []:
		s1 = 'Min'
		ax.scatter(minimum.x[0], minimum.x[1], marker='o', c='r')
		ax.text(minimum.x[0], minimum.x[1], s1, fontsize=12)


	fig.colorbar(im,ax=ax)

	plt.pause(0.1)#(0.0001)



def bubbles(x, y, z, f, narea):
    
    # f - set color
    # narea - set size 
    
    def zeroToOne(self):
        return (self-self.min())/(self.max()-self.min())
    
    def doCrosses(x, y, z, f, area, cond, mark, color):
        assemble_ = [x, y, z, f, area]
        assemble = pd.concat(assemble_, axis=1, sort=False)
        assemble = assemble[assemble.iloc[:,-1:] < cond]    

        [xn, yn, zn, fn, arean] = assemble
        ax.scatter(assemble[xn],assemble[yn],assemble[zn],
                   marker=mark,color=color)

    
    #plots bubbles
    area = zeroToOne(narea)
    area.name = narea.name
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_zlabel(z.name)
    
    #fig.suptitle(['color: ',f.name, ' size: ', narea.name]) 

    pcm = ax.scatter(x, y, z, c=f, s=1000*area, cmap="RdBu_r", alpha=0.4)
    # RdBu_r
    # nipy_spectral
    cmap = fig.colorbar(pcm, ax=ax)
    cmap.set_label(f.name)
    
    # plots crosses
    cond = 0.5
    doCrosses(x, y, z, f, area, cond, 'x', 'k')    
    #doCrosses(x, y, z, f, area, cond, '+', 'r') 
    #plots lim f
    

def plotOptm(w):

	[Nx, Ny, ne, E, h, a, b] = funVar()
