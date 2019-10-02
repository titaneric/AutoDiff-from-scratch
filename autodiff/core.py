from collections import defaultdict

import numpy as np

primitive_vhp = defaultdict(dict)

def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func][i] = downstream

if __name__ == "__main__":
    pass