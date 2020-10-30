
import matplotlib.pyplot as plt
import numpy as np

from function import function
from helpFun import funVar


def optim():
	# w0 = f(ne,E,h,a,b)

	[Nx, Ny, ne, E, h, a, b] = funVar()

	w = function([a/2, b/2])

	print(w)


	plt.ion()
	plt.scatter(E,w)
	plt.pause(0.1)



