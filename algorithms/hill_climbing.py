import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state

EPS = 1e-9


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        best_state = self.problem.generate_initial_state()
        best_cost = self.problem.evaluate(best_state)
        self.history.append(best_cost)

        for run in range(self.max_restarts + 1):
            current = self.problem.generate_initial_state() if run == 0 else random_state(self.problem)
            current_cost = self.problem.evaluate(current)
            visited = {current}
            sideways = 0
            if current_cost < best_cost:
                best_state, best_cost = current, current_cost

            while True:
                candidates = [n for n in get_neighbors(current, self.problem) if n not in visited]
                if not candidates:
                    break
                costs = [self.problem.evaluate(n) for n in candidates]
                lowest = min(costs)
                if lowest < current_cost - EPS:
                    sideways = 0
                elif abs(lowest - current_cost) <= EPS and sideways < self.max_sideways:
                    sideways += 1
                else:
                    break
                ties = [n for n, c in zip(candidates, costs) if c <= lowest + EPS]
                current = random.choice(ties)
                current_cost = self.problem.evaluate(current)
                visited.add(current)
                if current_cost < best_cost:
                    best_state, best_cost = current, current_cost
                self.history.append(best_cost)

        return best_state, best_cost