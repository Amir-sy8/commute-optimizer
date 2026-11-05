import random
from typing import List
from core.state import State
from core.problem import CommuteProblem


def simple_paths(network, start, goal, banned, limit):
    result = []

    def dfs(node, path):
        if len(result) >= limit:
            return
        if node == goal:
            result.append(tuple(path))
            return
        nxt = network.get_neighbors(node)
        random.shuffle(nxt)
        for v in nxt:
            if v not in path and v not in banned:
                path.append(v)
                dfs(v, path)
                path.pop()

    dfs(start, [start])
    return result


def random_state(problem: CommuteProblem) -> State:
    path = simple_paths(problem.network, problem.origin, problem.destination, set(), 1)[0]
    return State(path, random.choice(problem.get_valid_time_domain()))


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:
    neighbors = []
    seen = {state}

    def add(candidate):
        if candidate not in seen and problem.is_valid_state(candidate):
            seen.add(candidate)
            neighbors.append(candidate)

    for dt in (-problem.t_step, problem.t_step):
        add(State(state.path, round(state.departure_time + dt, 2)))

    path = state.path
    for i in range(len(path) - 1):
        for j in range(i + 1, len(path)):
            banned = set(path[:i]) | set(path[j + 1:])
            for sub in simple_paths(problem.network, path[i], path[j], banned, 20):
                add(State(path[:i] + sub + path[j + 1:], state.departure_time))

    return neighbors


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:
    shared = [n for n in parent1.path[1:-1] if n in parent2.path[1:-1]]
    random.shuffle(shared)
    avg = (parent1.departure_time + parent2.departure_time) / 2.0
    time = min(problem.get_valid_time_domain(), key=lambda t: abs(t - avg))

    for node in shared:
        i = parent1.path.index(node)
        j = parent2.path.index(node)
        path = parent1.path[:i] + parent2.path[j:]
        if problem.is_valid_path(path):
            return State(path, time)

    options = get_neighbors(parent1, problem)
    return random.choice(options) if options else parent1