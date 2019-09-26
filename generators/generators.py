try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose

import numpy as np
import itertools as it
from collections import defaultdict


class KnapsackGenerator:
    @staticmethod
    def generate(seed, number_of_items_types=10,
                 max_item_count=5, max_weight_per_item=25,
                 max_value_per_item=10, max_weight_pct=0.35):
        np.random.seed(seed)
        weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
        values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
        problem = mlrose.KnapsackOpt(length=number_of_items_types,
                                     maximize=True, max_val=max_item_count,
                                     weights=weights, values=values,
                                     max_weight_pct=max_weight_pct,
                                     multiply_by_max_item_count=False)
        return problem


class TSPGenerator:
    @staticmethod
    def generate(seed, number_of_cities, area_width=250, area_height=250):
        np.random.seed(seed)
        x_coords = np.random.randint(area_width, size=number_of_cities)
        y_coords = np.random.randint(area_height, size=number_of_cities)

        coords = list(tuple(zip(x_coords, y_coords)))
        duplicates = TSPGenerator.list_duplicates_(coords)

        while len(duplicates) > 0:
            for d in duplicates:
                x_coords = np.random.randint(area_width, size=len(d))
                y_coords = np.random.randint(area_height, size=len(d))
                for i in range(len(d)):
                    coords[d[i]] = (x_coords[i], y_coords[i])
                    pass
            duplicates = TSPGenerator.list_duplicates_(coords)
        distances = TSPGenerator.get_distances(coords, False)

        return mlrose.TSPOpt(coords=coords, distances=distances, maximize=False)


    @staticmethod
    def get_distances(coords, truncate=True):
        distances = [(c1, c2, np.linalg.norm(np.subtract(coords[c1], coords[c2])))
                     for c1, c2 in it.product(range(len(coords)), range(len(coords)))
                     if c1 != c2 and c2 > c1]
        if truncate:
            distances = [(c1, c2, int(d)) for c1, c2, d in distances]
        return distances

    #  https://stackoverflow.com/a/5419576/40410
    @staticmethod
    def list_duplicates_(seq):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return list((indices[1:] for _, indices in tally.items() if len(indices) > 1))


if __name__=='__main__':
    results = TSPGenerator.generate(123, 22)
    print(results)