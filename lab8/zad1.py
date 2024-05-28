import pygad

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


def funkcja_celu(solution, przedmioty):
    wartosc = 0
    waga = 0
    for i in range(len(przedmioty)):
        if solution[i] == 1:  # Jeśli przedmiot został wybrany
            wartosc += przedmioty[i][1]
            waga += przedmioty[i][2]
    if waga > 25:
        wartosc = (waga - 25) * 10 # Odrzucamy rozwiązania, które przekraczają łączną wagę 25
    return wartosc

def funkcja_oceny(pygad,solution, ga_instance):
    return funkcja_celu(solution, przedmioty)

# Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=4,
                       sol_per_pop=10,
                       num_genes=len(przedmioty),
                       fitness_func=funkcja_oceny,
                       mutation_percent_genes=10)

# Uruchomienie algorytmu genetycznego
ga_instance.run()

# Pobranie najlepszego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsza kombinacja:", solution)
print("Wartość najlepszej kombinacji:", solution_fitness)