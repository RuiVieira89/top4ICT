import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from function import function
from helpFun import funVar
from plotSurf import bubbles
from plotSurf import plotSurf
from plotSurf import visualize_corr

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():

	#plotSurf([], [])

	[Nx, Ny, ne, E, h, a, b] = funVar()
	w, sigma_max = function([a/2, b/2])

	# DataFrame
	x = 1000*h
	y = E/1e6
	z = ne
	f = 1000*w
	narea = sigma_max/1e6

	columns = ['Thickness [mm]', 'Young Modulus [MPa]', 'Poisson Ratio', 
		'Bending [mm]', 'Max Tension [MPa]']

	assemble = pd.DataFrame( data=np.transpose([x, y, z, f, narea]), columns=columns)

	assemble_ = assemble.loc[
	(assemble['Bending [mm]'] < 30) & (assemble['Max Tension [MPa]']<1000)]	

	print(f"{bcolors.HEADER}\nmaximum bend ={assemble_['Bending [mm]'].max():.2f}mm " + 
		f"minimum bend ={assemble_['Bending [mm]'].min():.2f}mm{bcolors.ENDC}")
	print(f"{bcolors.HEADER}\nmaximum Tension ={assemble_['Max Tension [MPa]'].max():.2f}MPa " + 
		f"minimum tension ={assemble_['Max Tension [MPa]'].min():.2f}MPa{bcolors.ENDC}")

	bubbles(assemble_.iloc[:,0], assemble_.iloc[:,1], assemble_.iloc[:,2], 
		assemble_.iloc[:,3], assemble_.iloc[:,4])

	visualize_corr(assemble_)


	plt.show()

	input('\nEnd')
	


if __name__ == "__main__":

	main()
