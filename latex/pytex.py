
from pytexit import py2tex

a = ['Dconst = (E*h**3)/(12*(1-ne**2))', 'k_ = (k*b**4)/(Dconst*np.pi**4)', 
'del_T = (T*alfa*Dconst*(1+ne)*np.pi**2)/b**2', 
'Wmn = (b**4)/(Dconst*np.pi**4)*(qmn + del_T*(m**2 * s**2 + n**2))/((m**2 * s**2 + n**2)**2 + k_)', 
'w0 = w0 + Wmn*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b)', 
'sigma_max = (6*qmn*2*b**2)/(np.pi**2*h**2*(s**2+1)**2)*(s**2+ne)']

for i in range(len(a)):
	py2tex(a[i])
	print('\n')

