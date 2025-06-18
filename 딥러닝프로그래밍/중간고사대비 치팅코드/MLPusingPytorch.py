import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

# 1. 데이터 준비
iris = load_iris()
X_np = iris['data']  # (150, 4)
Y_np = iris['target']  # (150,)

num_classes = np.max(Y_np) + 1 #클래스 수 자동계산
Y_onehot_np = np.eye(num_classes)[Y_np]

#random seed 설정
torch.manual_seed(0)

# tensor 변환
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.long)  # 정확도 측정용
Y_onehot = torch.tensor(Y_onehot_np, dtype=torch.float32)

# 2. MLP 모델 정의 (torch.nn 사용)
class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=5, output_dim=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) #은닉층
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 출력층 
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))  # 출력도 sigmoid (MSE 손실에 맞춤)
        return x

# 3. 모델 초기화
model = MLP(input_dim=4, hidden_dim=10, output_dim=3)

# 4. 손실 함수 & 옵티마이저 설정
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# 5. Mini-Batch 학습
def train(model, X, Y_onehot, batch_size=10, epochs=1000):
    num_samples = X.size(0)
    loss_history = []
    
    for epoch in range(epochs):
        # 미니배치 셔플
        # indices = np.random.permutation(num_samples)
        indices = torch.randperm(num_samples)
        X = X[indices]
        Y_onehot = Y_onehot[indices]

        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y_onehot[i:i+batch_size]

            # 순전파
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            # 역전파
            # 오류역전파 알고리즘을 수행하기 전에 이전 parameter의 gradient값을 0으로 초기화함
            # 그렇지 않은 경우 batch 단위 gradient가 계속 누적됨
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            loss_history.append(loss.item())
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")
    
    return loss_history

#학습 시작 시간 측정
start_time = time.time()

loss_history = train(model, X, Y_onehot, batch_size=20, epochs=5000)

#학습 후 시간 측정
end_time = time.time()

#학습 소요시간 계산 및 출력
elapsed_time = end_time - start_time
print(f"소요 시간: {elapsed_time:.4f}초")

# 6. 예측 및 정확도 평가
with torch.no_grad():
    outputs = model(X)
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds == Y).float().mean().item()
    print(f"\nClassification Accuracy: {accuracy:.4f}")

# 틀린 샘플 출력
incorrect = (preds != Y).nonzero(as_tuple=False).squeeze()
print("Incorrect predictions:")
if incorrect.shape != torch.Size([]):
   for idx in incorrect:
        print(f"Index: {idx.item()}, Sample: {X[idx].numpy()}, Pred: {preds[idx].item()}, True: {Y[idx].item()}")

# 7. 손실 그래프
plt.plot(loss_history)
plt.xlabel('Epoch (x10)')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)
plt.show()