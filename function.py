

import numpy as np

def function(x):

	return x

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


def platesTheory():
	# SOURCE:
	# Theory and Analysis of Elastic Plates and Shells, 
	# J. N. Reddy (pag.222)
	# Navier Solutions

	ne = 0.355 # Poisson's ratio (copper) 			PARAMETER
	E = 117e9 # elastcity module [Pa] (copper)  	PARAMETER
	h = 5/1000 # plate thickness [m] 				PARAMETER
	a = 100/1000 # plate x-length [m] 				PARAMETER
	b = 100/1000 # plate y-length [m] 				PARAMETER
	s = b/a
	alfa = 111e-6 # thermal diffusivity [m/s]
	Q0 = 10 # point force [N]

	m = 1
	n = 1


	## Rigidity matrix
	Dconst = (E*h**3)/(12*(1-ne**2))
	D = np.eye(6)
	D[0,0] = D[1,1] = Dconst
	D[0,1] = ne*Dconst
	D[-1,-1] = (1-ne)*Dconst/2

	k = 0 # spring coeff [N/m]
	k_ = (k*b**4)/(Dconst*np.pi**4) 

	## Thermal stresses
	T = 0 # Temperature [K]
	del_T = (T*alfa*Dconst*(1+ne)*np.pi**2)/b**2

	qmn = 4*Q0/a*b

	Wmn = (b**4)/(Dconst*np.pi**4)*(qmn + del_T*(m**2 * s**2 + n**2))/((m**2 * s**2 + n**2)**2 + k_)

	print(Wmn)



	''' ===================== SCRAP ===================== '''

	# Isotropic quasistatic Kirchhoff-Love plates

	'''ne = 0.355 # Poisson's ratio (copper)
	E = 117e9 # elastcity module ·[Pa] (copper)
	n = 3

	sigma = np.zeros(n) # tension
	sigma_old = np.zeros(n)
	strain = np.zeros(n) # strain

	# material matrix
	C = np.eye(3) 
	C[0,1] = C[0,1] = ne
	C[-1,-1] = 1 - ne
	C *= E/(1-ne**2)

	cond = False
	while cond is False:
		sigma = np.linalg.solve(C,strain)
		print(sigma)

		cond = sigma - sigma_old < 1e6'''