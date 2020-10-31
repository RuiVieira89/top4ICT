
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from function import function
from helpFun import funVar
from plotSurf import bubbles
from plotSurf import plotSurf

def main():

	#plotSurf([], [])

	[Nx, Ny, ne, E, h, a, b] = funVar()
	w, sigma_max = function([a/2, b/2])
	
	# DataFrame
	x = pd.DataFrame(h)
	x.name = 'Thickness [m]'
	y = pd.DataFrame(E)
	y.name = 'Young Modulus [Pa]'
	z = pd.DataFrame(ne)
	z.name = 'Poisson Ratio'
	f = pd.DataFrame(w)
	f.name = 'Bending [m]'
	narea = pd.DataFrame(sigma_max)
	narea.name = 'Max Tension [Pa]'

	print(h)

	bubbles(x, y, z, f, narea)
	plt.show()
	input('\nEnd')
	


if __name__ == "__main__":
	main()