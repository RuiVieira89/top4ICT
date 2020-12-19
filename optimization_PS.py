
from pymoo.algorithms.so_pattern_search import PatternSearch
from pymoo.factory import get_problem
from pymoo.optimize import minimize

from pymoo.model.problem import Problem

import warnings
warnings.simplefilter(action='ignore')

from function_opt import function
from helpFun import funVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

## Density data
HOME_FOLDER = os.getcwd()
LATEX_DIR = '/home/rui/dev/top4ICT/latex/'

df_common = pd.read_csv(HOME_FOLDER+'/data/Common_materials.tsv', sep='\t')
df_mat_dens = pd.read_csv(HOME_FOLDER+'/data/materials_strength_density.tsv', sep='\t')
df = pd.merge(df_common, df_mat_dens, on="Material")

MaterialDensity = df['Density low_y'].loc[45]

class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=3, 
			n_obj=2,
			n_constr=2,
			xl=np.array([0.25, 0.25, 0.001]),
			xu=np.array([0.5, 0.5, 0.01]),
			# elementwise_evaluation=True
			)

	def _evaluate(self, X, out, *args, **kwargs):

		w, sigma_max = function(X) 
		sysMass = np.prod(X, axis=1)*MaterialDensity

		out["F"] = np.column_stack([w, sysMass])
		out["G"] = np.column_stack([-w,-sysMass])

problem = MyProblem()

algorithm = PatternSearch(
	explr_delta=0.1, 
	explr_rho=0.1, 
	pattern_step=2, 
	eps=1e-12, 
	# display=PatternSearchDisplay()
	)

from pymoo.util.termination.default import MultiObjectiveDefaultTermination

termination = MultiObjectiveDefaultTermination(
    x_tol=1e-8,
    cv_tol=1e-6,
    f_tol=0.0025,
    nth_gen=5,
    n_last=30,
    n_max_gen=1000,
    n_max_evals=100000
)

from pymoo.optimize import minimize

res = minimize(
	problem, 
	algorithm, 
	termination, 
	seed=1, 
	save_history=True, 
	verbose=True
	)


''' === Object-Oriented Interface === '''

import copy

# perform a copy of the algorithm to ensure reproducibility
obj = copy.deepcopy(algorithm)

# let the algorithm know what problem we are intending to solve and provide other attributes
obj.setup(problem, termination=termination, seed=1)

# until the termination criterion has not been met
while obj.has_next():

	# perform an iteration of the algorithm
	obj.next()

	# access the algorithm to print some intermediate outputs
	print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F').min(axis=0)}")

# finally obtain the result object
result = obj.result()


from pymoo.visualization.scatter import Scatter


''' === Convergence === '''

n_evals = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation

# iterate over the deepcopies of algorithms
for algorithm in res.history:

	# store the number of function evaluations
	n_evals.append(algorithm.evaluator.n_eval)

	# retrieve the optimum from the algorithm
	opt = algorithm.opt

	# store the least contraint violation in this generation
	cv.append(opt.get("CV").min())

	# filter out only the feasible and append
	feas = np.where(opt.get("feasible"))[0]
	_F = opt.get("F")[feas]
	F.append(_F)


''' === Hypvervolume (HV) === '''

import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume

# MODIFY - this is problem dependend
ref_point = np.array([1.0, 1.0])

# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in F]

fig = plt.figure()
# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
# fig.savefig(LATEX_DIR + 'convergence.eps', format='eps')
plt.show()



from pymoo.factory import get_visualization, get_decomposition

F = res.F
weights = np.array([0.01, 0.99])
decomp = get_decomposition("asf")

# We apply the decomposition and retrieve the best value (here minimum):
I = get_decomposition("asf").do(F, weights).argmin()
print("Best regarding decomposition: Point %s - %s" % (I, F[I]))

plot = get_visualization("scatter")
plot.add(F, color="blue", alpha=0.5, s=30)
plot.add(F[I], color="red", s=40)
plot.do()
# plot.apply(lambda ax: ax.arrow(0, 0, F[I][0], F[I][1], color='black',
# 	head_width=0.001, head_length=0.001, alpha=0.4))
plot.show()


## Parallel Coordinate Plots
from pymoo.visualization.pcp import PCP
plotPCP = PCP().add(res.F).add(res.F[I], color='r').show()


print(f'A solução ideal é encontrada para a deformação {res.F[I][0]:.4f}m e massa {res.F[I][0]:.4f}kg com os parametros a {res.X[I][0]:.4f} b {res.X[I][1]:.4f} e h {res.X[I][2]:.4f}')



#print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))