
import numpy as np 

def funVar():
	#Discretization

	Nx = 10 
	Ny = 10

	'''ne = 0.33 # Poisson's ratio  					PARAMETER
	E = 6.9e10 # elastcity module [Pa]   			PARAMETER
	h = 5/1000 # plate thickness [m] 				PARAMETER
	a = 300/1000 # plate x-length [m] 				PARAMETER
	b = 300/1000 # plate y-length [m] 				PARAMETER'''

	f =0.9 # final point
	ne = np.arange(0.1,f+f/Nx,f/Nx) # Poisson's ratio  					PARAMETER
	f = 0.9e11
	E = np.arange(1e10,f+f/Nx,f/Nx) # elastcity module [Pa]   			PARAMETER
	f = 10/1000
	h = np.arange(1/1000,f+f/Nx,f/Nx) # plate thickness [m] 				PARAMETER
	f = 100/100
	a = np.arange(10/100,f+f/Nx,f/Nx) # plate x-length [m] 				PARAMETER
	b = np.arange(10/100,f+f/Nx,f/Nx) # plate y-length [m] 				PARAMETER

	A = np.array( np.meshgrid(ne, E, h, a, b) )
	A = np.reshape(A, (A.shape[0], A.shape[1]*A.shape[2]*A.shape[3]*A.shape[4]*A.shape[4]))


	return Nx, Ny, A #Nx, Ny, ne, E, h, a, b

