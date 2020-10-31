
import numpy as np 
import pandas as pd 

def funVar():

	HOME_FOLDER = '/home/rui/dev/top4ICT'

	#df_phonon = pd.read_csv(HOME_FOLDER+'/data/phonon_dielectric/phonon_dielectric_mp.csv')

	df_common = pd.read_csv(HOME_FOLDER+'/data/Common_materials.tsv', sep='\t')
	df_mat_dens = pd.read_csv(HOME_FOLDER+'/data/materials_strength_density.tsv', sep='\t')


	df = pd.merge(df_common, df_mat_dens, on="Material")

	df_use = df[['Material', 
		'Category_x',
		'Young Modulus low', 'Young Modulus high',
		'Density low_y', 'Density high_y',
		'Resistivity low', 'Resistivity high',
		'Yield Strength low', 'Yield Strength high']]

	df_use['Young Modulus low'] = df_use['Young Modulus low']*1e6
	df_use['Young Modulus high'] = df_use['Young Modulus high']*1e6
	df_use['Yield Strength low'] = df_use['Yield Strength low']*1e6
	df_use['Yield Strength high'] = df_use['Yield Strength high']*1e6

	#Discretization
	Nx = df_use.shape[0]
	Ny = df_use.shape[0]


	f =0.9 # final point
	ne = np.arange(0.1,f+f/Nx,f/Nx) # Poisson's ratio  					PARAMETER
	# Young module [Pa]   			PARAMETER	
	E = np.array((df_use['Young Modulus high'] - df_use['Young Modulus low'])/2) 

	f = 10/1000
	h = np.arange(1/1000,f+f/Nx,f/Nx) # plate thickness [m] 				PARAMETER
	#f = 100/100
	#a = np.arange(10/100,f+f/Nx,f/Nx) # plate x-length [m] 				PARAMETER
	#b = np.arange(10/100,f+f/Nx,f/Nx) # plate y-length [m] 				PARAMETER
	a = 300/1000
	b = 300/1000

	A = np.array( np.meshgrid(*[ne, E], h) ) # mix the variables
	A = np.reshape(A, (A.shape[0], -1))


	return Nx, Ny, A[0], A[1], A[2], a, b

	'''return: Nx, Ny, ne, E, h, a, b'''
