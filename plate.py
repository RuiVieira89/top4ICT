
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from function import function
from helpFun import funVar
from plotSurf import bubbles
from plotSurf import plotSurf


def main():

	#plotSurf([], [])

	[Nx, Ny, ne, E, h, a, b] = funVar()
	w, sigma_max = function([a/2, b/2])

	# DataFrame
	x = pd.DataFrame(1000*h)
	x.name = 'Thickness [mm]'
	y = pd.DataFrame(E/1e6)
	y.name = 'Young Modulus [MPa]'
	z = pd.DataFrame(ne)
	z.name = 'Poisson Ratio'
	f = pd.DataFrame(1000*w)
	f.name = 'Bending [mm]'
	narea = pd.DataFrame(sigma_max/1e6)
	narea.name = 'Max Tension [MPa]'

	bubbles(x, y, z, f, narea)
	plt.show()
	input('\nEnd')
	


if __name__ == "__main__":

	main()
