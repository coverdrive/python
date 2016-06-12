from __future__ import absolute_import

# This functions operate on irregularly-shaped structures as well.
# When irregularly shaped, transpose_list_of_dicts and transpose_list_of_lists
# will "compress" its results and so they won't transpose back to the original
# irregularly shaped input. On regularly shaped (rectangular) inputs, all these
# operations are 1 to 1, i.e.,  transpose of transpose will give back the original
# regularly shaped input.


def transpose_dict_of_dicts(d):
    """
    Returns the transposed dictionary of dictionaries.
    Works on irregularly shaped (non-rectangular) dicts of dicts
    """
    all_y = set(y for _, di in d.iteritems() for y, _ in di.iteritems())
    return {y: {x: val for x, di in d.iteritems() for y1, val in di.iteritems() if y1 == y} for y in all_y}

def transpose_dict_of_lists(d):
    """
    Returns the transposed list of dictionaries.
    Works on irregularly shaped (non-rectangular) dicts of lists
    """
    max_len = max(len(l) for _, l in d.iteritems())
    return [{k: l[i] for k, l in d.iteritems() if i < len(l)} for i in xrange(max_len)]

def transpose_list_of_dicts(l):
    """
    Returns the transposed dictionary of lists.
    Works on irregularly shaped (non-rectangular) lists of dicts
    Will 'compress' the result on irregularly shaped input
    """
    all_k = set(k for d in l for k, _ in d.iteritems())
    return {k: [val for d in l for k1, val in d.iteritems() if k1 == k] for k in all_k}

def transpose_list_of_lists(l):
    """
    Returns the transposed list of lists.
    Works on irregularly shaped (non-rectangular) lists of lists
    Will 'compress' the result on irregularly shaped input
    """
    max_len = max(len(lin) for lin in l)
    return [[lin[i] for lin in l if i < len(lin)] for i in xrange(max_len)]

import unittest

class TestTranspose(unittest.TestCase):

    def runTest(self):
        d = {
                's1': {'t1': 1.0, 't2': 0.9, 't3': 0.7},
                's2': {'t2': 0.9, 't4': 1.0},
                's3': {'t1': 1.0},
                's4': {'t3': 0.3, 't5': 0.9}
        }

        d_T = transpose_dict_of_dicts(d)

        d_T_T = transpose_dict_of_dicts(d_T)

        print "d = ", d
        print "d Transpose = ", d_T
        print "d Transpose Transpose = ", d_T_T
        print d == d_T_T

        d1 = {
                's1': [4, 3, 9, 9, 8, 1],
                's2': [6, 2],
                's3': [4, 1, 9, 0, 8],
                's4': [8, 1, 0, 6, 4, 9, 8]
        }

        d1_T = transpose_dict_of_lists(d1)

        d1_T_T = transpose_list_of_dicts(d1_T)

        print "d1 = ", d1
        print "d1 Transpose = ", d1_T
        print "d1 Transpose Transpose = ", d1_T_T
        print d1 == d1_T_T

        l = [
            {'t1': 1.0, 't2': 0.9, 't3': 0.7, 't4': 1.0},
            {'t2': 0.9, 't6': 1.0},
            {'t1': 1.0, 't5': 1.1, 't7': 0.0},
            {'t3': 0.3}
        ]

        l_T = transpose_list_of_dicts(l)

        print "l = ", l
        print "l Transpose = ", l_T

        l1 = [
            [3, 4, 1, 9],
            [8, 9, 10, 16, 2, 1, 4],
            [3, 2],
            [4, 5, 9, 8, 7],
            [6],
            [4, 8, 7]
        ]

        l1_T = transpose_list_of_lists(l1)

        print "l1 = ", l1
        print "l1 Transpose = ", l1_T

if __name__ == '__main__':
    unittest.main()
