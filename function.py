
import numpy as np

from helpFun import funVar

def function(X):
	# SOURCE:
	# Theory and Analysis of Elastic Plates and Shells, 
	# J. N. Reddy (pag.222)
	# Navier Solutions

	[Nx, Ny, A] = funVar()
	ne = A[0]
	E = A[1]
	h = A[2]
	a = A[3]
	b = A[4]

	x = X[0]
	y = X[1]
	s = b/a
	alfa = 111e-6 # thermal diffusivity [m/s]
	Q0 = 3000 # point force [N]

	m = -1
	n = -1

	## Rigidity matrix
	Dconst = (E*h**3)/(12*(1-ne**2))
	'''D = np.eye(6)
	D[0,0] = D[1,1] = Dconst
	D[0,1] = ne*Dconst
	D[-1,-1] = (1-ne)*Dconst/2'''

	k = 0 # spring coeff [N/m]
	k_ = (k*b**4)/(Dconst*np.pi**4) 

	## Thermal stresses
	T = 0 # Temperature [K]
	del_T = (T*alfa*Dconst*(1+ne)*np.pi**2)/b**2

	qmn = 4*Q0/a*b

	w0 = 0
	w0_old = 0
	cond = False
	while cond == False:

		m += 2
		n += 2

		Wmn = (b**4)/(Dconst*np.pi**4)*(qmn + del_T*(m**2 * s**2 + n**2))/ \
		((m**2 * s**2 + n**2)**2 + k_)

		w0 = w0 + Wmn*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)
		
		cond = np.nanmax((w0 - w0_old)/w0) < 1e-8
		w0_old = w0.copy()


	return w0


def Encurvadura(x):

	'''Encurvadura'''
	l = x[0] # 0.4 # heigth [m]
	e = x[1] # 0.001 # force eccentricity[m], 0 if none 
	P = 10 # force [N]
	E = 117e9 # elastcity module ·[Pa] (copper)
	r = 1/100 # [m]
	I = (np.pi/4)*r**4
	k = np.sqrt(np.abs(P)/(E*I))

	flexa = e*(1-np.cos(k*l))/np.cos(k*l) # flexion [m]
	Mf_max = P*(e+flexa) # maximum moment of flextion
	W = l*(np.pi*r**2) # volume [mn]
	tension_max = P/(np.pi*r**2) + Mf_max/W


	return tension_max


