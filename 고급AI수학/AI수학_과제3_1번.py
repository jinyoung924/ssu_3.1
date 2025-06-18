import numpy as np
import matplotlib.pyplot as plt

# 시드 고정 (재현 가능한 결과)
np.random.seed(0)

# 입력 데이터 50개 생성
x = np.random.rand(50)

# 참값 선형 모델 (정답 선): y = 0.5x + 0.8
target_y = 0.5 * x + 0.8

# 관측 데이터: 정답 선 + 가우시안 노이즈(표준편차 0.25)
noise = np.random.normal(0, 0.25, size=50)
y = target_y + noise

# 모델 파라미터 초기화
weight = 0.0
bias = 0.0

# 학습 설정
learning_rate = 0.25
max_iterations = 50

# 경사하강법을 이용한 학습
for i in range(1, max_iterations + 1):
    prediction = weight * x + bias        # 현재 모델의 예측값
    error = prediction - y                # 오차 = 예측값 - 실제 관측값

    # 평균 제곱 오차(MSE)의 gradient 계산
    dw = (2 / len(x)) * np.dot(error, x)
    db = (2 / len(x)) * np.sum(error)

    # 가중치와 절편 업데이트
    weight -= learning_rate * dw
    bias -= learning_rate * db

    # 현재 MSE 계산
    mse = np.mean(error ** 2)

    # 학습 중간 과정 출력 (처음, 10회 간격, 마지막)
    if i == 1 or i % 10 == 0 or i == max_iterations:
        print(f"[{i:2d}회차] MSE: {mse:.4f}, weight: {weight:.4f}, bias: {bias:.4f}")

# 결과 시각화
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label='Observed Data', color='skyblue')  # 실제 데이터
plt.plot(x, target_y, 'k--', label='Target Line: y = 0.5x + 0.8')  # 정답 선
plt.plot(x, weight * x + bias, 'r-', label=f'Fitted Line: y = {weight:.2f}x + {bias:.2f}')  # 모델이 학습한 선
plt.title('Gradient Descent Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()