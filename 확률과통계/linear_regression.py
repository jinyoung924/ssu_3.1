import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find the optimal w and b
def grad_descent(xs, ys, w_init, b_init, alpha: float, num_steps: float):
    w_hat = w_init
    b_hat = b_init  
    w_hats = [w_hat]
    b_hats = [b_hat]
    for i in range(num_steps):
        # Compute Jacobian (J)
        Jw = compute_Jw(xs, ys, w_hat, b_hat)
        Jb = compute_Jb(xs, ys, w_hat, b_hat)
        w_hat -= alpha * Jw
        b_hat -= alpha * Jb
        w_hats.append(w_hat)
        b_hats.append(b_hat)             
    return w_hats, b_hats

def compute_Jw(xs, ys, w, b):
    Jw = -2 * np.sum(xs * ((ys - (w*xs + b))), axis=0)
    return Jw

def compute_Jb(xs, ys, w, b):
    Jb = -2 * np.sum(((ys - (w*xs + b))), axis=0)    
    return Jb

# Ground truth (GT)
np.random.seed(0)
x = np.random.uniform(low=-1, high=1, size=30)
w = 2
b = 1
y = w * x + b

# Observation (+ Gaussian noise)
yo = y + np.random.normal(0, 0.1, 30)

# Initial models
w_init = float(np.random.randn(1))
b_init = float(np.random.randn(1))
y_init = w_init * x + b_init

# Linear regression
alpha = 0.01
num_steps = 30
w_hats, b_hats = grad_descent(x, yo, w_init, b_init, alpha=alpha, num_steps=num_steps)
y_hats = w_hats[-1] * x + b_hats[-1]
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.grid()
ax.plot(x, yo, '.')
ax.plot(x, y, c='g')
ax.plot(x, y_init, c='r')
ax.plot(x, y_hats, c='b')
ax.set_title("Linear regression")
print(w_hats[::5])
print(b_hats[::5])
plt.show()