import math
import matplotlib.pyplot as plt
import pygad
import numpy as np

def endurance(x, y, z, u, v, w):
 return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

ga_instance = pygad.GA(num_generations=64,
                       num_parents_mating=4,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=6,  
                       gene_type=float,
                       gene_space=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                       initial_population= np.random.rand(10, 6),
                       mutation_percent_genes=20)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepszy fitnes =", solution_fitness)
print("Najlepsze parametry : ", solution)

fitness_values = ga_instance.best_solutions_fitness
plt.plot(fitness_values)
plt.xlabel('Generacja')
plt.ylabel('Fitness')
plt.grid()
plt.savefig("plot2.png")
plt.show()