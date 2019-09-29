from mlrose.opt_probs import FlipFlopOpt, QueensOpt, MaxKColorOpt

try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose

import numpy as np
import itertools as it
import networkx as nx
from collections import defaultdict


class KnapsackGenerator:
    @staticmethod
    def generate(seed, number_of_items_types=10,
                 max_item_count=5, max_weight_per_item=25,
                 max_value_per_item=10, max_weight_pct=0.35,
                 multiply_by_max_item_count=True):
        np.random.seed(seed)
        weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
        values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
        problem = mlrose.KnapsackOpt(length=number_of_items_types,
                                     maximize=True, max_val=max_item_count,
                                     weights=weights, values=values,
                                     max_weight_pct=max_weight_pct,
                                     multiply_by_max_item_count=multiply_by_max_item_count)
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


class FlipFlopGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        problem = FlipFlopOpt(length=size)
        return problem


class QueensGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        problem = QueensOpt(length=size)
        return problem


class MaxKColorGenerator:
    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None):

        """
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        """
        np.random.seed(seed)
        # all nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                      n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([_ for _ in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = MaxKColorOpt(edges=edges, length=number_of_nodes, max_colors=max_colors)
        return problem


"""
class NNWeightGenerator:
    @staticmethod
    def generate(seed, hidden_layer_sizes, activation, algorithm, max_attempts):
        np.random.seed(seed)
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=hidden_layer_sizes,
                                         activation=activation,
                                         algorithm=algorithm,
                                         bias=True,
                                         is_classifier=True,
                                         early_stopping=True,
                                         max_attempts=max_attempts)
        problem = QueensOpt(length=size)
        return problem
"""

if __name__=='__main__':
    results = TSPGenerator.generate(123, 22)
    print(results)
