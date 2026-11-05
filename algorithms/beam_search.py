from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        beam = [self.problem.generate_initial_state()]
        for _ in range(self.beam_width * 5):
            if len(beam) >= self.beam_width:
                break
            candidate = random_state(self.problem)
            if candidate not in beam:
                beam.append(candidate)

        self.history.append(min(self.problem.evaluate(s) for s in beam))

        for _ in range(self.max_iterations):
            pool = list(beam)
            for state in beam:
                pool.extend(get_neighbors(state, self.problem))
            pool = list(dict.fromkeys(pool))
            pool.sort(key=self.problem.evaluate)
            new_beam = pool[:self.beam_width]
            self.history.append(self.problem.evaluate(new_beam[0]))
            if set(new_beam) == set(beam):
                break
            beam = new_beam

        best_state = min(beam, key=self.problem.evaluate)
        return best_state, self.problem.evaluate(best_state)