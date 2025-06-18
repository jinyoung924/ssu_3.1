import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

x = st.uniform.rvs(0, 100000, size=10)
m = np.mean(x)
s = np.sqrt(np.var(x, ddof=1))  # 표본 표준편차

n = len(x)
a = 0.05
k = st.t.ppf(1 - a / 2, n - 1)  # t값
u = m + k * s / np.sqrt(n)      # 상한
l = m - k * s / np.sqrt(n)      # 하한
print(k, l, u)

print(st.t.interval(1-a, n-1, loc=m, scale=s/np.sqrt(n)))
