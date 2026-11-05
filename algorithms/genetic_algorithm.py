import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)
        self.tournament_size: int = self.params.get("tournament_size", 3)

    def fitness(self, state: State) -> float:
        return 1.0 / (1.0 + self.problem.evaluate(state))

    def select(self, population):
        contenders = random.sample(population, min(self.tournament_size, len(population)))
        return max(contenders, key=self.fitness)

    def solve(self) -> Tuple[State, float]:
        population = [self.problem.generate_initial_state()]
        while len(population) < self.population_size:
            population.append(random_state(self.problem))

        best_state = min(population, key=self.problem.evaluate)
        best_cost = self.problem.evaluate(best_state)
        self.history.append(best_cost)

        for _ in range(self.generations):
            ranked = sorted(population, key=self.problem.evaluate)
            next_population = ranked[:min(self.elitism, self.population_size)]

            while len(next_population) < self.population_size:
                parent1 = self.select(population)
                parent2 = self.select(population)
                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2, self.problem)
                else:
                    child = parent1
                if random.random() < self.mutation_rate:
                    options = get_neighbors(child, self.problem)
                    if options:
                        child = random.choice(options)
                next_population.append(child)

            population = next_population
            generation_best = min(population, key=self.problem.evaluate)
            generation_cost = self.problem.evaluate(generation_best)
            if generation_cost < best_cost:
                best_state, best_cost = generation_best, generation_cost
            self.history.append(best_cost)

        return best_state, best_cost