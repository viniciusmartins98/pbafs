import random
import os
import numpy as np

def randint(a, b, seed_bytes=128):
    random.seed(os.urandom(seed_bytes))
    return random.randint(a, b)

def uniform(a, b, seed_bytes=128):
    random.seed(os.urandom(seed_bytes))
    return random.uniform(a, b)