import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수 및 파생 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 오버플로우 방지

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# 소프트맥스 함수
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 원-핫 인코딩
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

# 다중 클래스 분류 데이터 생성 (더 복잡한 데이터)
np.random.seed(42)
n_samples_per_class = 50
X_list = []
y_list = []

# Class 0: 원점 주변
X0 = np.random.normal([1, 1], 0.3, (n_samples_per_class, 2))
X_list.append(X0)
y_list.extend([0] * n_samples_per_class)

# Class 1: 우상단
X1 = np.random.normal([3, 3], 0.4, (n_samples_per_class, 2))
X_list.append(X1)
y_list.extend([1] * n_samples_per_class)

# Class 2: 좌하단
X2 = np.random.normal([0.5, 3.5], 0.3, (n_samples_per_class, 2))
X_list.append(X2)
y_list.extend([2] * n_samples_per_class)

X = np.vstack(X_list)
y = np.array(y_list)
y_onehot = one_hot(y, 3)

# 1단계: 원본 데이터 산점도 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
colors = ['blue', 'red', 'green']
for class_idx in np.unique(y):
    plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1],
                color=colors[class_idx], label=f'Class {class_idx}',
                s=30, alpha=0.7, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original Training Data')
plt.legend()
plt.grid(True)

# 2단계: 신경망 구조 정의 (입력2 → 은닉7 → 출력3)
np.random.seed(42)
W1 = np.random.randn(2, 7) * 0.1  # 은닉층 노드 7개
b1 = np.zeros((1, 7))
W2 = np.random.randn(7, 3) * 0.1
b2 = np.zeros((1, 3))

# 하이퍼파라미터
lr = 1.0
max_iter = 1000
n_samples = X.shape[0]

print("신경망 학습 시작...")
print(f"은닉층 노드 수: 7개")
print(f"학습률: {lr}")
print(f"최대 반복수: {max_iter}")

# 3단계: 학습 루프 (편미분을 반복문으로 계산)
for iteration in range(max_iter):
    # 순전파
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    # 오차 계산
    error = a2 - y_onehot
    mse = np.mean(error ** 2)
    
    # 역전파: 출력층 그래디언트 (반복문으로 계산)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    dz1_total = np.zeros_like(z1)
    
    # 각 샘플에 대해 반복문으로 그래디언트 계산
    for i in range(n_samples):
        # 출력층 그래디언트
        dz2_i = error[i:i+1]  # (1, 3)
        dW2 += np.dot(a1[i:i+1].T, dz2_i)  # (7, 1) x (1, 3) = (7, 3)
        db2 += dz2_i
        
        # 은닉층 그래디언트 
        dz1_i = np.dot(dz2_i, W2.T) * sigmoid_deriv(z1[i:i+1])  # (1, 7)
        dz1_total[i:i+1] = dz1_i
    
    # 은닉층 가중치 그래디언트 (반복문으로 계산)
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    
    for i in range(n_samples):
        dW1 += np.dot(X[i:i+1].T, dz1_total[i:i+1])  # (2, 1) x (1, 7) = (2, 7)
        db1 += dz1_total[i:i+1]
    
    # 파라미터 업데이트
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    # 로그 출력
    if iteration % 2000 == 0:
        print(f"Iteration {iteration}, MSE: {mse:.4f}")

print("학습 완료!")

# 4단계: 결정 경계 시각화
plt.subplot(1, 2, 2)

# 그리드 생성
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                       np.linspace(x2_min, x2_max, 200))
grid = np.c_[xx1.ravel(), xx2.ravel()]

# 그리드에 대한 예측
z1_grid = np.dot(grid, W1) + b1
a1_grid = sigmoid(z1_grid)
z2_grid = np.dot(a1_grid, W2) + b2
a2_grid = softmax(z2_grid)

# 각 클래스별 확률
prob_0 = a2_grid[:, 0].reshape(xx1.shape)
prob_1 = a2_grid[:, 1].reshape(xx1.shape)
prob_2 = a2_grid[:, 2].reshape(xx1.shape)

# 원본 데이터 산점도
for class_idx in np.unique(y):
    plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1],
                color=colors[class_idx], label=f'Class {class_idx}',
                s=30, alpha=0.8, edgecolors='k')

# 결정 경계 등위곡선 (확률 0.5)
contour0 = plt.contour(xx1, xx2, prob_0, levels=[0.5], colors='blue', 
                      linestyles='dashed', linewidths=2)
contour1 = plt.contour(xx1, xx2, prob_1, levels=[0.5], colors='red', 
                      linestyles='dashed', linewidths=2)
contour2 = plt.contour(xx1, xx2, prob_2, levels=[0.5], colors='green', 
                      linestyles='dashed', linewidths=2)

# 등위곡선 라벨링
plt.clabel(contour0, fmt={0.5: 'C0:0.5'}, fontsize=8)
plt.clabel(contour1, fmt={0.5: 'C1:0.5'}, fontsize=8)
plt.clabel(contour2, fmt={0.5: 'C2:0.5'}, fontsize=8)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundaries (7 Hidden Nodes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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