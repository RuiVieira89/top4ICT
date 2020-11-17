
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
        assemble = pd.concat(assemble_, axis=1, sort=False, ignore_index=True)
        assemble = assemble[assemble.loc[:,4] < cond] 
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

    pcm = ax.scatter(x, y, z, c=f, s=100*area, cmap="RdBu_r", alpha=0.4)
    # RdBu_r
    # nipy_spectral
    cmap = fig.colorbar(pcm, ax=ax)
    cmap.set_label(f.name)
    
    # plots crosses
    cond = 0.5
    # doCrosses(x, y, z, f, area, cond, 'x', 'k')    
    # doCrosses(x, y, z, f, area, cond, '+', 'r') 
    # plots lim f


def visualize_corr(df): # correlation between all parameters
    
    '''    
    Pearson's correlation is a measure of the linear relationship between two continuous random 
    variables. It does not assume normality although it does assume finite variances and finite 
    covariance. When the variables are bivariate normal, Pearson's correlation provides a complete 
    description of the association.
    Spearman's correlation applies to ranks and so provides a measure of a monotonic relationship 
    between two continuous random variables. It is also useful with ordinal data and is robust to 
    outliers (unlike Pearson's correlation).
    The distribution of either correlation coefficient will depend on the underlying distribution, 
    although both are asymptotically normal because of the central limit theorem.
    Kendall rank correlation: Kendall rank correlation is a non-parametric test that measures the 
    strength of dependence between two variables.  If we consider two samples, a and b, where each 
    sample size is n, we know that the total number of pairings with a b is n(n-1)/2.  The following 
    formula is used to calculate the value of Kendall rank correlation:
    use Pearson because of:
    http://d-scholarship.pitt.edu/8056/
    '''

    
    
    df_corr = df.corr(method='pearson')
    
    data1 = df_corr.values
    data1 = np.around(data1, decimals=2)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    
    for i in range(len(data1)):
        for j in range(len(data1)):
            text = ax1.text(j, i, data1[i, j], 
                            horizontalalignment='left', 
                            verticalalignment='top', color="k")              

    
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    #plt.show()


def plotOptm(w):

	[Nx, Ny, ne, E, h, a, b] = funVar()
