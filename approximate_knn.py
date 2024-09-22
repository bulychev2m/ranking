from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm
import random, time


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # pointA : (1, D), documets : (N, D) -> (N, 1)
    return np.linalg.norm(documents - pointA, axis=1, keepdims=True)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    """
    для каждой точки мы сначала вычисляем расстояния до всех точек (или до sampling_share от всех точек):
        выбираем из них топ самых дальних num_candidates_for_choice_long и из них случайно выбираем num_edges_long точек.
        для ближайших точек тоже самое

    Returns
    -------
    Dict[int, List[int]]
        ключ - индекс точки (количество ключей равно количеству точек N), 
        значение - список индексов точек, которые образуют связи в виде ребер 
        (длинных и коротких совместно, без разделения).
    """
    graph = defaultdict(list)
    N = data.shape[0]

    # считаем расстояния между любыми двумя точками (всего надо посчитать n*(n-1)/2 расстояний, n - число точек)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        distance_matrix[i, i+1:] = dist_f(data[i], data[i+1:]).flatten()
        distance_matrix[i+1:, i] = distance_matrix[i, i+1:]

    for first_point in tqdm(range(N)):
        # Извлечение расстояний для первой точки
        distances_to_others = distance_matrix[first_point]
        
        # Исключаем текущую точку (саму себя)
        valid_points = np.array([j for j in range(N) if j != first_point])
        valid_distances = distances_to_others[valid_points]

        # Если используется выборка, выбираем случайные точки для расчетов
        if use_sampling:
            sample_size = int(N * sampling_share)
            random_sample_idxs = random.sample(list(valid_points), sample_size)
            points_ = np.array(random_sample_idxs)
            distances_arr = valid_distances[np.isin(valid_points, random_sample_idxs)]
        else:
            points_ = valid_points
            distances_arr = valid_distances
        
        # Находим индексы самых ближних и дальних точек
        partitioned_indices = np.argpartition(distances_arr, [num_candidates_for_choice_short - 1, -num_candidates_for_choice_long])
        
        farthest_points = points_[partitioned_indices[-num_candidates_for_choice_long:]]
        farthest = np.random.choice(farthest_points, size=num_edges_long, replace=False)

        closest_points = points_[partitioned_indices[:num_candidates_for_choice_short]]
        closest = np.random.choice(closest_points, size=num_edges_short, replace=False)

        graph[first_point].extend(farthest)
        graph[first_point].extend(closest)

    ans = {first_point: list(set(graph[first_point])) for first_point in graph}
    return ans


def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    pass
        
    




def main():
    data = np.random.rand(10000, 32)
    ans = create_sw_graph(data)
    print(ans)

if __name__ == "__main__":
    # test_main()
    start_time = time.time()
    main()
    print(f"Программа работала {time.time() - start_time} секунд")