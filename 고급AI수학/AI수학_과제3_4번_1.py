import numpy as np
import matplotlib.pyplot as plt

# 1단계: 다중 클래스 분류 데이터 생성 및 시각화
X = np.array([
    [0, 0],  # class 0 (파랑)
    [1, 0],  # class 1 (빨강)
    [0, 1],  # class 1 (빨강)
    [1, 1]   # class 2 (초록)
])
y = np.array([0, 1, 1, 2])  # 클래스 레이블

# 원-핫 인코딩
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

Y = one_hot(y, 3)

# 시각화 색
colors = ['blue', 'red', 'green']

# 2단계: 신경망 구현
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 파라미터 초기화
np.random.seed(0)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 3)
b2 = np.zeros((1, 3))

lr = 1.0
max_iter = 10000

# 학습
for i in range(max_iter):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    error = a2 - Y
    mse = np.mean(error ** 2)
    
    dz2 = error
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    dz1 = np.dot(dz2, W2.T) * sigmoid_deriv(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if i % 2000 == 0:
        print(f"Iteration {i}, MSE: {mse:.4f}")

# 3단계: 결정경계 시각화
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return np.argmax(a2, axis=1)

def plot_decision_boundary(X, labels, predict_func):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
    for class_idx in np.unique(labels):
        plt.scatter(X[labels == class_idx, 0], X[labels == class_idx, 1],
                    color=colors[class_idx], label=f'Class {class_idx}',
                    s=100, edgecolors='k')
    plt.legend()
    plt.title("MLP Decision Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, predict)