import pygad
import numpy as np

#a

przedmioty = [
    ["zegar",100,7],
    ["obraz-pejzaż",300,7],
    ["obraz-portret",200,6],
    ["radio",40,2],
    ["laptop",500,5],
    ["lampka nocna",70,6],
    ["srebrne sztućce",100,1],
    ["porcelana",250,3],
    ["figura z brazu",300,10],
    ["skorzana torebka",280,3],
    ["odkurzacz",300,15]
]

#bc

def funkcja_celu(solution, przedmioty):
    wartosc = 0
    waga = 0
    for i in range(len(przedmioty)):
        wartosc += solution[i]*przedmioty[i][1]
        waga += solution[i]*przedmioty[i][2]
    if waga > 25:
        wartosc = 0 # Odrzucamy rozwiązania, które przekraczają łączną wagę 25
    return wartosc

def funkcja_oceny(pygad,solution, ga_instance):
    return funkcja_celu(solution, przedmioty)

def func(ga_instance): 
    if(ga_instance.best_solution()[1] >= 1630):
        ga_instance.run_completed = True

# Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(
    gene_space=[0,1],
    num_generations=100,
    num_parents_mating=4,
    sol_per_pop=10,
    num_genes=len(przedmioty),
    fitness_func=funkcja_oceny,
    mutation_percent_genes=20,
    keep_parents=1,
    mutation_type="random",
    on_generation=func
    )

# Uruchomienie algorytmu genetycznego
ga_instance.run()

# Pobranie najlepszego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsza kombinacja:", solution)
print("Wartość najlepszej kombinacji:", solution_fitness)

wybrane_przedmioty = [przedmioty[i] for i in range(len(solution)) if solution[i] == 1]
wybrane_nazw = [item[0] for item in wybrane_przedmioty]
total_value = sum(item[1] for item in wybrane_przedmioty)
total_weight = sum(item[2] for item in wybrane_przedmioty)

print("Najlepsze przedmioty: ", wybrane_nazw)
print("Ich wartość: ", total_value)
print("Waga: ", total_weight)

ga_instance.plot_fitness()
ga_instance.summary()