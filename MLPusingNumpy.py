import numpy as np
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def initialize_weights(input_dim, hidden_dim, output_dim):
    U1 = np.random.randn(hidden_dim, input_dim + 1) * 0.01  # +1 for bias
    U2 = np.random.randn(output_dim, hidden_dim + 1) * 0.01  # +1 for bias
    return U1, U2

def train_mlp(X, Y, hidden_dim=10, lr=0.1, batch_size=16, epochs=100):
    num_samples, input_dim = X.shape
    output_dim = Y.shape[1]  # Y는 one-hot 형태
    loss_history = [] #loss값의 추이를 그래프를 표현하기 위한 리스트
    
    # Bias 추가를 위한 입력 확장
    z0=np.ones((num_samples, 1))
    X = np.hstack([z0, X])  # x0 = 1
    #X = np.concatenate([np.ones((num_samples, 1)), X],axis=1) # x0 = 1
    

    # 가중치 초기화
    U1, U2 = initialize_weights(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        # 무작위 셔플
        indices = np.random.permutation(num_samples)
        X, Y = X[indices], Y[indices]

        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]

            dU1 = np.zeros_like(U1)
            dU2 = np.zeros_like(U2)

            for x, y in zip(X_batch, Y_batch):
                x = x.reshape(-1, 1)  # (d+1, 1)
                y = y.reshape(-1, 1)  # (c, 1)

                # 순전파
                zsum = U1.dot(x)               # (p, 1)
                z = sigmoid(zsum)           # 은닉층 출력 (p, 1)
                z = np.vstack(([[1]], z))     # z[0] = 1 추가 → (p+1, 1)
                #z = np.concatenate(([[1]], z),axis=0)     # z0 = 1 추가 → (p+1, 1)
                osum = U2.dot(z)               # (c, 1)
                o = sigmoid(osum)           # 출력층

                # 역전파
                delta = (y - o) * sigmoid_deriv(osum)  # (c, 1)
                dU2 += -delta.dot(z.T)                    # (c, p+1)

                eta = ((U2.T).dot(delta))[1:] * sigmoid_deriv(zsum)  # (p, 1)
                dU1 += -eta.dot(x.T)


            # 평균 그레디언트 적용
            U2 -= lr * (dU2 / batch_size)
            U1 -= lr * (dU1 / batch_size)

        # 에포크별 손실 출력 (옵션)
        if (epoch + 1) % 10 == 0:
            y_pred_all = predict(X, U1, U2)
            #MSE 손실함수
            loss = np.mean((Y - y_pred_all) ** 2) 
            #Cross Entropy 손실함수
            #loss = -np.mean(np.sum(Y * np.log(y_pred_all + 1e-8), axis=1))
            loss_history.append(loss)
            print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

    return U1, U2, loss_history


def predict(X, U1, U2):
    num_samples = X.shape[0]
    if X.shape[1] == U1.shape[1] - 1:
        X = np.hstack([np.ones((num_samples, 1)), X])
    zsum = X @ U1.T
    z = sigmoid(zsum)
    z = np.hstack([np.ones((num_samples, 1)), z])
    osum = z @ U2.T
    o = sigmoid(osum)
    return o


iris = load_iris()

X = iris['data']
Y = iris['target']
num_classes = np.max(Y) + 1 #클래스 수 자동계산

Y_onehot = np.eye(num_classes)[Y]

np.random.seed(0)

#num= 0
#for x,y in zip(X[shuffle_idx],Y_hot[shuffle_idx]):
#    print("input data[{0}]:{1},target:{2}".format(num,x,y))
#    num +=1

# 학습 실행
U1, U2, loss_history = train_mlp(X, Y_onehot, hidden_dim=5, lr=0.5, batch_size=10, epochs=1000)

# 예측 결과
Y_pred = predict(X, U1, U2)
pred_labels = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_onehot, axis=1)

#accuracy 측정
accuracy = np.mean(pred_labels == Y_true)
print("Classification Accuracy for Iris dataset:{0}".format(accuracy))

# 틀린 샘플을 찾아  출력
diff_idx = np.where(Y_true != pred_labels)
diff_idx = np.array(diff_idx)
diff_idx = diff_idx.reshape(-1)
print(diff_idx)

print("Incorredtly predicted samples\n")
for idx in diff_idx:
    print("Index:{0}, Sample:{1}, Predicted Label:{2}, True Label:{3}\n".\
        format(idx, X[idx], pred_labels[idx], Y_true[idx]))

# loss 그래프 출력
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.show()
