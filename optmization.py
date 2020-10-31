
import matplotlib.pyplot as plt
import numpy as np

from function import function
from helpFun import funVar


def optim():

	[Nx, Ny, ne, E, h, a, b] = funVar()

	w = function([a/2, b/2])

	plt.ion()
	plt.scatter(E,w)
	plt.pause(0.1)



