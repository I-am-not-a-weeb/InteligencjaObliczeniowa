import matplotlib.pyplot as plt
import random

from aco import AntColony


COORDS = (
(59, 23), (74, 17), (43, 50),  (20, 84), (70, 65), (29, 90), (87, 83),
(35, 40), (65, 73), (43, 85),  (73, 23), (42, 41), (42, 43), (33, 67),   
(70, 13), (79, 14), (100, 65), (81, 43), (7, 9),   (98, 91), (37, 40), 
(48, 40), (98, 45), (45, 81),  (20, 52), (43, 11), (3, 91),  (81, 78),  
(86, 81), (30, 59)
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=500, alpha=0.3, beta=5, 
                    pheromone_evaporation_rate=0.70, pheromone_constant=1000.0,
                    iterations=50)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.savefig("py2.png")
plt.show()