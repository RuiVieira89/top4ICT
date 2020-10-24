
import matplotlib.pyplot as plt
import numpy as np

from function import function
from helpFun import funVar


def optim():
	# w0 = f(ne,E,h,a,b)

	[Nx, Ny, A] = funVar()
	ne = A[0]
	E = A[1]
	h = A[2]
	a = A[3]
	b = A[4]

	w = function([a/2, b/2])


	plt.ion()
	plt.scatter(E,w)
	plt.pause(0.1)

	print(A.shape)
	print(w.shape)


