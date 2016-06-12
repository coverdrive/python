from itertools import product, combinations, permutations


def get_cartesian_product(tuple_of_sets):
    return frozenset(product(*tuple_of_sets))


def get_combinations(given_set, k):
    return frozenset(frozenset(x) for x in combinations(given_set, k))


def get_permutations(given_set, k):
    return frozenset(permutations(given_set, k))
