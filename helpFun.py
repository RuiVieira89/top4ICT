
def funVar():
	#Discretization
	Nx = 10 
	Ny = 10

	ne = 0.355 # Poisson's ratio (copper) 			PARAMETER
	E = 117e9 # elastcity module [Pa] (copper)  	PARAMETER
	h = 1/1000 # plate thickness [m] 				PARAMETER
	a = 100/1000 # plate x-length [m] 				PARAMETER
	b = 100/1000 # plate y-length [m] 				PARAMETER

	return Nx, Ny, ne, E, h, a, b

