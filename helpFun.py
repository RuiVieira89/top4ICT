
def funVar():
	#Discretization

	Nx = 10 
	Ny = 10

	ne = 0.33 # Poisson's ratio  					PARAMETER
	E = 6.9e10 # elastcity module [Pa]   			PARAMETER
	h = 5/1000 # plate thickness [m] 				PARAMETER
	a = 300/1000 # plate x-length [m] 				PARAMETER
	b = 300/1000 # plate y-length [m] 				PARAMETER

	return Nx, Ny, ne, E, h, a, b

