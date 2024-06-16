import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher 


options1 = {'c1':0.5, 'c2':0.3, 'w':0.5} 
optimizer1 = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options1)

options2 = {'c1':0.2, 'c2':0.6, 'w':0.5} 
optimizer2 = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options2)
optimizer1.optimize(fx.sphere, iters=50) 
optimizer2.optimize(fx.sphere, iters=50) 
# tworzenie animacji 
m = Mesher(func=fx.sphere) 
animation = plot_contour(pos_history=optimizer1.pos_history, mesher=m, mark=(0, 0))
animation.save('plot0.gif', writer='imagemagick', fps=10)
animation = plot_contour(pos_history=optimizer2.pos_history, mesher=m, mark=(0, 0))
animation.save('plot1.gif', writer='imagemagick', fps=10)