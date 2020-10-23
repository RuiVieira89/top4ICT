

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
	# J. N. Reddy (pag.261)


	
	w0 = (Wh + qn)*sin(beta_n*y)



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