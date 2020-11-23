
import warnings
warnings.simplefilter(action='ignore')

from function_opt import function

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=3,
			n_obj=2,
			n_constr=2,
			xl=np.array([0.25,0.25,0.001]),
			xu=np.array([0.5,0.5,0.01]))

	def _evaluate(self, x, out, *args, **kwargs):
		w, sigma_max = function(x) 

		out["F"] = np.column_stack([w])
		out["G"] = np.column_stack([sigma_max])


problem = MyProblem()

POP = 100
algorithm = NSGA2(pop_size=POP)

from pymoo.factory import get_termination

termination = get_termination("n_gen", 100)

res = minimize(problem,
	algorithm,
	termination,
	("n_gen", POP),
	verbose=True,
	seed=1)

plot = Scatter()
plot.add(res.pop.get("X"), color="red")
# plot.add(res.X, color="red")
plot.show()
