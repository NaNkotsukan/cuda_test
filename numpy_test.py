import numpy as np
import time

xp = np

for i in range(1, 5):
    for j in range(1, 10):
        n = 10**i*j
        a = xp.random.rand(n, n).astype(np.float32)
        start = time.time()
        b = xp.linalg.inv(a)
        t = time.time() - start
        print(n, t*1000)
