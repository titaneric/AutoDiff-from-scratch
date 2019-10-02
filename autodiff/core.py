from collections import defaultdict

import numpy as np
import networkx as nx


primitive_vhp = defaultdict(dict)

def register_vjp(func, vhp_list):
    print(func)
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func][i] = downstream

if __name__ == "__main__":
    pass