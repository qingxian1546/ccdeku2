import math
def entropy(probabilities):
    i = 0
    for p in probabilities:
        i += -p * math.log(p, 2)
    return i
probabilities = [0.5, 0.25, 0.125, 0.125]
a = entropy(probabilities)
print("x信息熵为:", a)
import math
def entropy2(probabilities):
    n = 0
    for p in probabilities:
        n += -p * math.log(p, 2)
    return n
probabilities = [0.25, 0.25, 0.25, 0.25]
b = entropy2(probabilities)
print("y信息熵为:", b)
import numpy as np
biaoge = np.array([[0.125, 0.0625, 0.03125,0.03125],
                        [0.0625, 0.125, 0.03125,0.03125],
                        [0.0625,0.0625, 0.0625,0.0625],
                        [0.25,0,0,0]])
jieguo = -np.sum(biaoge * np.log2(biaoge))
print("联合熵 H(X,Y) =", jieguo)

import numpy as np
c = np.array([[0.125, 0.0625, 0.03125,0.03125],
                        [0.0625, 0.125, 0.03125,0.03125],
                        [0.0625,0.0625, 0.0625,0.0625],
                        [0.25,0,0,0]])
marginal_prob_X = np.sum(c, axis=1)
entropy = 0
for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        if c[i, j] != 0:
            entropy -= c[i, j] * np.log(c[i, j] / marginal_prob_X[i])
print("条件熵 H(Y|X) 为:", entropy)
