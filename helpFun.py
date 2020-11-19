import numpy as np 
import pandas as pd 
import os

def funVar():

	HOME_FOLDER = os.getcwd()

	#df_phonon = pd.read_csv(HOME_FOLDER+'/data/phonon_dielectric/phonon_dielectric_mp.csv')
	df_common = pd.read_csv(HOME_FOLDER+'/data/Common_materials.tsv', sep='\t')
	df_mat_dens = pd.read_csv(HOME_FOLDER+'/data/materials_strength_density.tsv', sep='\t')

	df = pd.merge(df_common, df_mat_dens, on="Material")
	#df = df[df['Category_x']=='Metals']

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

	# Poisson's ratio  					PARAMETER
	ne = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 
	0.4, 0.25, 0.3, 0.001, 0.33, 0.31, 0.39, 0.35, 0.39, 0.33, 0.45, 0.43, 0.36, 
	0.46, 0.38, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.36, 0.2, 0.25, 0.35, 0.25,
	0.22, 0.44, 0.33, 0.27, 0.3, 0.34, 0.32, 0.29, 0.27, 0.25]) 

	# Young module [Pa]   				PARAMETER	
	E = np.array((df_use['Young Modulus high'] - df_use['Young Modulus low'])/2) 

	f = 10/1000
	h = np.arange(0.1/1000,f+f/Nx,f/Nx) # plate thickness [m] 				PARAMETER
	#f = 100/100
	#a = np.arange(10/100,f+f/Nx,f/Nx) # plate x-length [m] 				PARAMETER
	#b = np.arange(10/100,f+f/Nx,f/Nx) # plate y-length [m] 				PARAMETER
	a = 300/1000
	b = 300/1000

	# A = np.array( np.meshgrid(*[ne, E], h) ) # mix the variables
	# A = np.reshape(A, (A.shape[0], -1))

	return Nx, Ny, ne[45], E[45], h, a, b
	'''return: Nx, Ny, ne, E, h, a, b'''

	#==== Material index ====#
	# 0            Flexible Foam VLD
	# 1             Flexible Foam LD
	# 2             Flexible Foam MD
	# 3                Rigid Foam LD
	# 4                Rigid Foam MD
	# 5                Rigid Foam HD
	# 6                     Isoprene
	# 7                 Butyl rubber
	# 8                          EVA
	# 9                     Neoprene
	# 10                Polyurethane
	# 11         Silicone elastomers
	# 12                     Leather
	# 13             Wood (parallel)
	# 14        Wood (perpendicular)
	# 15                      Bamboo
	# 16                        Cork
	# 17                   Polyester
	# 18                     Epoxies
	# 19                        PMMA
	# 20                          PS
	# 21                          PA
	# 22                          PC
	# 23                          PE
	# 24                         PET
	# 25                          PP
	# 26                        PTFE
	# 27                        PEEK
	# 28                       Brick
	# 29                       Stone
	# 30                    Concrete
	# 31                       Glass
	# 32                        GFRP
	# 33                        CFRP
	# 34    Aluminun/silicon carbide
	# 35               Boron carbide
	# 36             Silicon nitride
	# 37             Silicon carbide
	# 38                     Alumina
	# 39            Tungsten carbide
	# 40                 Lead alloys
	# 41                   Cu alloys
	# 42                      Steels
	# 43                   Ni alloys
	# 44                   Ti Alloys
	# 45                   Al alloys
	# 46                   Mg alloys
	# 47                   Cast iron
	# 48                   Zn alloys
