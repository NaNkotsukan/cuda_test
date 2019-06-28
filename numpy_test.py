import numpy as np
import time

xp = np

for i in range(1, 4):
    for j in range(1, 10):
        n = 10**i*j
        a = xp.random.rand(n, n)
        start = time.time()
        b = xp.linalg.inv(a)
        t = time.time() - start
        print(n, t*1000)
