# TIPS:: https://github.com/msu-coinlab/pymoo/blob/master/doc/source/getting_started.ipynb
# DOCS:: https://pyomo.readthedocs.io/_/downloads/en/stable/pdf/

import warnings
warnings.simplefilter(action='ignore')

from function_opt import function
from helpFun import funVar

import matplotlib.pyplot as plt
import numpy as np
from pymoo.model.problem import Problem

# def function(x):
# 	return x[0]**2-x[1], x[0]-x[2]

class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=3, 
			n_obj=2,
			n_constr=2,
			xl=np.array([0.25,0.25,0.001]),
			xu=np.array([0.5,0.5,0.01]))

	def _evaluate(self, X, out, *args, **kwargs):
		w, sigma_max = function(X) 

		out["F"] = np.column_stack([w])
		out["G"] = np.column_stack([sigma_max])


problem = MyProblem()

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.algorithms.so_genetic_algorithm import GA

algorithm = NSGA2(
	pop_size=20,
	n_offsprings=10,
	sampling=get_sampling("real_random"),
	crossover=get_crossover("real_sbx", prob=0.9, eta=15),
	mutation=get_mutation("real_pm", eta=20),
	eliminate_duplicates=True
)

from pymoo.factory import get_termination

termination = get_termination("n_gen", 40)

from pymoo.optimize import minimize

res = minimize(problem, 
	algorithm, 
	termination, 
	seed=1, 
	save_history=True, 
	verbose=True)

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





''' === derived Pareto set === '''

from pymoo.visualization.scatter import Scatter

# get the pareto-set and pareto-front for plotting
ps = problem.pareto_set(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=False)

# Design Space
plot = Scatter(title = "Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
if ps is not None:
	plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.apply(lambda ax: ax.set_xlim(0, 0.5))
plot.apply(lambda ax: ax.set_ylim(0, 0.5))
plot.show()

exit(0)

# Objective Space
plot = Scatter(title = "Objective Space")
plot.add(res.F)
if pf is not None:
	plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.show()

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

# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()

''' === IGD === '''

import matplotlib.pyplot as plt
from pymoo.performance_indicator.igd import IGD

if pf is not None:

	# for this test problem no normalization for post prcessing is needed since similar scales
	normalize = False

	metric = IGD(pf=pf, normalize=normalize)

	# calculate for each generation the HV metric
	igd = [metric.calc(f) for f in F]

	# visualze the convergence curve
	plt.plot(n_evals, igd, '-o', markersize=4, linewidth=2, color="green")
	plt.yscale("log")          # enable log scale if desired
	plt.title("Convergence")
	plt.xlabel("Function Evaluations")
	plt.ylabel("IGD")
	plt.show()



