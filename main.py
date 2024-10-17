import numpy as np
import random
from tsplib95 import load

choosen_num_clusters = 40

class PSO:
    def __init__(self, num_particles, num_iterations, problem):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.graph = problem.get_graph()  # Obter a matriz de adjacência
        self.nodes = list(problem.get_nodes())  # Lista de nós
        self.clusters = self._define_clusters()  # Definir clusters
        self.num_clusters = len(self.clusters)
        self.global_best_position = None
        self.global_best_fitness = float("inf")

        # Inicializa as partículas
        self.particles = [self._create_particle() for _ in range(self.num_particles)]

    def _define_clusters(self):
        """Cria clusters manualmente."""
        random.shuffle(self.nodes)
        clusters = []
        cluster_size = len(self.nodes) // choosen_num_clusters
        for i in range(0, len(self.nodes), cluster_size):
            clusters.append(self.nodes[i : i + cluster_size])
        return clusters

    def _create_particle(self):
        position = [np.random.choice(cluster) for cluster in self.clusters]
        fitness = self._calculate_fitness(position)
        return {
            "position": position,
            "fitness": fitness,
            "best_position": position,
            "best_fitness": fitness,
            "velocity": np.random.rand(self.num_clusters),  # Inicializa a velocidade
        }

    def _calculate_fitness(self, position):
        total_distance = 0
        for i in range(len(position) - 1):
            total_distance += self.graph[position[i]][position[i + 1]]["weight"]
        total_distance += self.graph[position[-1]][position[0]][
            "weight"
        ]  # Fechar o tour
        return total_distance

    def optimize(self):
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                # Atualiza a posição
                particle["position"] = self._update_position(particle)

                # Atualiza o fitness
                particle["fitness"] = self._calculate_fitness(particle["position"])
                if particle["fitness"] < particle["best_fitness"]:
                    particle["best_position"] = particle["position"]
                    particle["best_fitness"] = particle["fitness"]

                # Atualiza a melhor global
                if particle["fitness"] < self.global_best_fitness:
                    self.global_best_position = particle["position"]
                    self.global_best_fitness = particle["fitness"]

            print(
                f"Iteration {iteration + 1}/{self.num_iterations}, Best Fitness: {self.global_best_fitness}"
            )

    def _update_position(self, particle):
        new_position = []
        for i in range(self.num_clusters):
            # Lógica aleatória para a atualização
            if self.global_best_position is not None and np.random.rand() < 0.5:
                new_position.append(self.global_best_position[i])
            else:
                new_position.append(particle["best_position"][i])
        return new_position


# Exemplo de uso
problem = load("d198.tsp")  # Insira o caminho para seu arquivo .tsp
pso = PSO(num_particles=30, num_iterations=100, problem=problem)
pso.optimize()
