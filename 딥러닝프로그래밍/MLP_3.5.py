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
            #-- 한 행씩 때어서 가져오는데 가져오면 행벡터다. 그래서 reshape으로 열벡터로 바꿔준다

                # 순전파
            zsum = U1.dot(x)               # (p, 1)
            z = sigmoid(zsum)           # 은닉층 출력 (p, 1)
            z = np.vstack(([[1]], z))     # z[0] = 1 추가 → (p+1, 1)
            #z = np.concatenate(([[1]], z),axis=0)     # z0 = 1 추가 → (p+1, 1)
            osum = U2.dot(z)               # (c, 1)
            o = sigmoid(osum)           # 출력층
        #--  zsum에 활성함수 적용하면 z
        #-- dot(x): 닷프로덕트 

            # 역전파
            delta = (y - o) * sigmoid_deriv(osum)  # (c, 1)
            #-- 출력의 오류값과 시그모이드 미분한 함수에 osum을 넣은 값 그걸 델타로 정의!

            dU2 = -delta.dot(z.T)                    # (c, p+1)
            #-- 

            eta = ((U2.T).dot(delta))[1:] * sigmoid_deriv(zsum)  # (p, 1)
            #-- 에타 :
            dU1 = -eta.dot(x.T)
            #-- 그레디언트값을 계속 누적해서 더함


            # 그레디언트 값에 러닝레이트를 곱해서 가중치를 업데이트
            U2 -= lr * (dU2)
            U1 -= lr * (dU1)
            #-- 가중치(U행렬)를 업데이트 시키는 부분
            #-- 한배치의 한번의 파라미터 업데이트가 이루어짐

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
