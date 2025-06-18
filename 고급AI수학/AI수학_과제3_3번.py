import numpy as np
import matplotlib.pyplot as plt

#1단계
# XOR 입력과 타겟 정의
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  # XOR 논리

# 시각화
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=100, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('[XOR] Training Data')
plt.grid(True)
plt.show()


#2단계
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 초기 가중치와 편향 설정 (은닉층 2개 노드)
np.random.seed(42)
W1 = np.random.randn(2, 2)   # 입력층 → 은닉층
b1 = np.zeros(2)
W2 = np.random.randn(2)      # 은닉층 → 출력층
b2 = 0.0

# 학습 설정
lr = 1
max_iter = 10000

# 학습 반복
for i in range(max_iter):
    # 순전파
    z1 = X @ W1 + b1      # 은닉층 입력
    a1 = sigmoid(z1)      # 은닉층 출력
    z2 = a1 @ W2 + b2     # 출력층 입력
    y_pred = sigmoid(z2)  # 최종 출력

    # 오차 계산
    error = y_pred - y

    # 역전파: 출력층 → 은닉층
    delta2 = error * sigmoid_deriv(z2)
    dW2 = a1.T @ delta2 / len(X)
    db2 = np.mean(delta2)

    delta1 = (np.outer(delta2, W2)) * sigmoid_deriv(z1)
    dW1 = X.T @ delta1 / len(X)
    db1 = np.mean(delta1, axis=0)

    # 가중치 및 편향 업데이트
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # 학습 로그 출력
    if i % 2000 == 0:
        mse = np.mean(error**2)
        print(f"[{i:5d} iter] MSE: {mse:.4f}")


#3단계
# 테스트용 격자 생성
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 순전파를 통해 격자 전체의 예측값 계산
z1 = grid @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
zz = sigmoid(z2).reshape(xx.shape)

# 결정경계 시각화
plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=100, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('[XOR] Decision Boundary')
plt.grid(True)
plt.show()

