import numpy as np
import matplotlib.pyplot as plt

# [단계 1] AND 논리 데이터 정의
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])  # AND 연산 결과

# [단계 2-1] 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# [단계 2-2] 모델 파라미터 초기화
w = np.zeros(2)  # 가중치 벡터 (w1, w2)
b = 0.0          # 바이어스
learning_rate = 1
max_iterations = 10000

# [단계 2-3] 경사하강법(GD)을 이용한 파라미터 업데이트
costs = []
for i in range(max_iterations):
    z = np.dot(X, w) + b            # 선형 결합: z = w1*x1 + w2*x2 + b
    y_pred = sigmoid(z)             # 예측 확률
    error = y_pred - y              # 예측 오차

    # MSE 비용 함수 (Mean Squared Error)
    cost = np.mean((error) ** 2)
    costs.append(cost)

    # 경사(gradient) 계산
    dw = np.dot(X.T, error) / len(X)
    db = np.sum(error) / len(X)

    # 파라미터 업데이트
    w -= learning_rate * dw
    b -= learning_rate * db

# 학습된 파라미터 출력
print(f"✅ 학습 완료된 가중치 w: {w}")
print(f"✅ 학습 완료된 바이어스 b: {b}")

# [단계 3] 결과 시각화 - 데이터 포인트와 결정 경계 그리기

plt.figure(figsize=(6, 5))

# 데이터 포인트 그리기
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k', label="Data")

# 결정 경계 선 그리기
x1_vals = np.linspace(-0.1, 1.1, 100)
x2_vals = -(w[0] * x1_vals + b) / w[1]  # w1*x1 + w2*x2 + b = 0 → x2 = -(w1*x1 + b)/w2
plt.plot(x1_vals, x2_vals, 'g--', label="Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("AND Logic Classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# [옵션] 비용 함수 시각화
plt.figure()
plt.plot(costs)
plt.title("Cost Function (MSE) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE Cost")
plt.grid(True)
plt.show()