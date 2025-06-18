import numpy as np
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#-- 시그모이드 함수

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)
#-- 시그모이드 함수의 미분갑

def initialize_weights(input_dim, hidden_dim, output_dim):
    U1 = np.random.randn(hidden_dim, input_dim + 1) * 0.01  # +1 for bias
    U2 = np.random.randn(output_dim, hidden_dim + 1) * 0.01  # +1 for bias
    return U1, U2
#-- U1,U2를 난수 발생으로 초기화 해준다

def train_mlp(X, Y, hidden_dim=10, lr=0.1, batch_size=16, epochs=100):
    num_samples, input_dim = X.shape
    output_dim = Y.shape[1]  # Y는 one-hot 형태
    loss_history = [] #loss값의 추이를 그래프를 표현하기 위한 리스트

    #-- 아이리스데이터 셋의 xy를 가져옴 --> x:입력, y: onehot 인코딩된 값, lr= learning rate

    # Bias 추가를 위한 입력 확장
    z0=np.ones((num_samples, 1)) 
    X = np.hstack([z0, X])  # x0 = 1 인 행을 병합
    #X = np.concatenate([np.ones((num_samples, 1)), X],axis=1) # x0 = 1

    #-- X는 150x4 행렬 거기에 바이어스노드인 x0=1 을 추가함
    #-- z0=np.ones((num_samples, 1)) : 값이 1인 바이어스노드를 만들고 , 다음 코드에서 병합.

    # 가중치 초기화
    U1, U2 = initialize_weights(input_dim, hidden_dim, output_dim)


#-- 에포크 시작

    for epoch in range(epochs):
        # 훈련집합 XY를 같은 순서로 무작위 셔플
        indices = np.random.permutation(num_samples) #--indice 행렬을 이용해서 셔플을 하겠다. 쓰는이유 1.원본은 유지 2. 섞는 순서를 같게 해주려고
        X, Y = X[indices], Y[indices] #-- X와 Y를 같은 순서로 섞은 새로운 행렬을 만듬 (여기서는 원본을 유지하지 않지만, 다른 코드에서는 유지하는 경우가 많다)

#-- 배치 구현을 뺴기 

        for x, y in zip(X, Y):
            x = x.reshape(-1, 1)  # (d+1, 1) 
            y = y.reshape(-1, 1)  # (c, 1)
        #-- 한행씩 때어오면 행벡터라 reshape으로 열벡터로 바꾸기

            # 순전파
            zsum = np.zeros((U1.shape[0], 1))
            for j in range(U1.shape[0]):
                for i in range(U1.shape[1]):
                    zsum[j] += U1[j][i] * x[i]

            z = np.zeros((U1.shape[0] + 1, 1))  # z0 포함
            z[0] = 1  # bias
            for j in range(1, z.shape[0]):
                z[j] = sigmoid(zsum[j - 1])

            osum = np.zeros((U2.shape[0], 1))
            for k in range(U2.shape[0]):
                for j in range(U2.shape[1]):
                    osum[k] += U2[k][j] * z[j]

            o = np.zeros_like(osum)
            for k in range(o.shape[0]):
                o[k] = sigmoid(osum[k])

            # 역전파
            delta = np.zeros_like(o)
            for k in range(delta.shape[0]):
                delta[k] = (y[k] - o[k]) * sigmoid_deriv(osum[k])

            dU2 = np.zeros_like(U2)
            for k in range(U2.shape[0]):
                for j in range(U2.shape[1]):
                    dU2[k][j] = -delta[k] * z[j]

            eta = np.zeros((U1.shape[0], 1))
            for j in range(U1.shape[0]):
                sum_term = 0
                for k in range(U2.shape[0]):
                    sum_term += delta[k] * U2[k][j + 1]
                eta[j] = sigmoid_deriv(zsum[j]) * sum_term

            dU1 = np.zeros_like(U1)
            for j in range(U1.shape[0]):
                for i in range(U1.shape[1]):
                    dU1[j][i] = -eta[j] * x[i]

            # 가중치 업데이트
            for k in range(U2.shape[0]):
                for j in range(U2.shape[1]):
                    U2[k][j] -= lr * dU2[k][j]
            for j in range(U1.shape[0]):
                for i in range(U1.shape[1]):
                    U1[j][i] -= lr * dU1[j][i]

#-- 배치 구현 끝

        # 에포크별 손실 출력 (옵션)
        if (epoch + 1) % 10 == 0:
            y_pred_all = predict(X, U1, U2)
           #MSE 손실함수
            loss = np.mean((Y - y_pred_all) ** 2) 
           #Cross Entropy 손실함수
           #loss = -np.mean(np.sum(Y * np.log(y_pred_all + 1e-8), axis=1))
            loss_history.append(loss) #-- 나중에 확인하려고 history에 저장
            print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

    return U1, U2, loss_history

#-- 
def predict(X, U1, U2):
    num_samples = X.shape[0]
    if X.shape[1] == U1.shape[1] - 1:
        X = np.hstack([np.ones((num_samples, 1)), X])
    zsum = X @ U1.T #-- @ 계산자
    z = sigmoid(zsum)
    z = np.hstack([np.ones((num_samples, 1)), z])
    osum = z @ U2.T
    o = sigmoid(osum)
    return o

#-- 아이리스데이터셋 정보
iris = load_iris()

X = iris['data']
Y = iris['target']
num_classes = np.max(Y) + 1 #클래스 수 자동계산
# 브레이크 포인트 : f9키 , f5 : 디버깅 모드에서 컴파일 시작

Y_onehot = np.eye(num_classes)[Y]

np.random.seed(0)

#num= 0
#for x,y in zip(X[shuffle_idx],Y_hot[shuffle_idx]):
#    print("input data[{0}]:{1},target:{2}".format(num,x,y))
#    num +=1

#학습 전 시간 측정
start_time = time.time()

# 학습 실행
U1, U2, loss_history = train_mlp(X, Y_onehot, hidden_dim=10, lr=0.5, batch_size=20, epochs=5000)
#-- 여기서 히든 디멘션에서는 바이어스 노드도 포함

#학습 후 시간 측정
end_time = time.time()

#학습 소요시간 계산 및 출력
elapsed_time = end_time - start_time
print(f"소요 시간: {elapsed_time:.4f}초")

# 예측 결과
Y_pred = predict(X, U1, U2)
pred_labels = np.argmax(Y_pred, axis=1)#-- 프레딕션레이블: 출력으로 원핫인코딩 값을 레이블로 바꿈, 010 = 2번째 값이 제일 큼-> 인덱스 값인 1이 레이블 값이 됨.
Y_true = np.argmax(Y_onehot, axis=1)

#accuracy 측정
accuracy = np.mean(pred_labels == Y_true)#-- 프레딕션레이블 값과 정답 값이 같은 걸 mean 계산 -> 정확도 측정
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
