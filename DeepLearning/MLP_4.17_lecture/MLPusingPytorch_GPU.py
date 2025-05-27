import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import time

# 0. GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 데이터 준비
iris = load_iris()
X_np = iris['data']  # (150, 4)     #-- 엔디 array
Y_np = iris['target']  # (150,)     #-- ground truth(부류) 의 레이블 값 (0,1,2)
#-- 150개의 자료, 4차원의 정보(꽃받침의 길이, 꽃받침 넓이, 꽃잎의 길이, 꽃잎의 넓이) -> 이걸 세개의 붓꽃(0,1,2)으로 분류할거다.

num_classes = np.max(Y_np) + 1  # 클래스 수 자동계산
Y_onehot_np = np.eye(num_classes)[Y_np]     #-- (150 x 3)구조의 원핫인코딩 방식(맞는 값만 1이고 나머지는 0인 구조)

#random seed 설정
torch.manual_seed(0)

# tensor 변환 및 GPU로 이동
X = torch.tensor(X_np, dtype=torch.float32,device=device)
Y = torch.tensor(Y_np, dtype=torch.long,device=device)      #-- 학습할 때 쓰는건 아니고 정확도 측정용이다.
Y_onehot = torch.tensor(Y_onehot_np, dtype=torch.float32,device=device).to(device)

# 2. MLP 모델 정의 (torch.nn 사용)
#-- nn.module : 레이어(층)에 관련된 클래스를 지원 , cost function도 지원
class MLP(nn.Module):       #-- nn.module을 부모함수로 MLP 함수를 정의(오버라이드)
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):       #-- __init__ : 초기설정을 정하는 함수
        super(MLP, self).__init__()     
        #-- super : 부모클래스를 초기화하는 함수 = child 클래스의 이름을 인자로 넘겨서 저 부모클래스를 초기화해줌 -> 그러면 부모함수의 변수와 함수를 100% 활용할 수 있음)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 은닉층
        #-- input_dim: 입력층에 주어지는 데이터샘플 : 4 (바이어스노드를 포함하지 않은 차원, 파이토치에서 내부적으로 바이어스노드가 추가됨/ numpy에서는 직접 넣어줬지만)
        #-- hidden_dim: 사용자가 몇개를 쓸지 정해야 한다(하이퍼 파라미터)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 출력층 
        #-- hidden_dim : 내가 히든노드를 5개로 설정햇으면 파이토치가 내부적으로 바이어스 노드를 더해서 6개로 만들어준다
        #-- output_dim : data set에 따라 정해지는 값, 여기서는 붓꽃을 세개로 분류하니깐 값이 3이 된다.
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):       #-- forward : 모델의 계산의 정의하는 함수
        x = self.act1(self.fc1(x))      #-- x :은닉층의 입력값(zsum벡터, x에 활성함수 씌우면 z벡터)
        x = self.act2(self.fc2(x))  # 출력도 sigmoid (MSE 손실에 맞춤)
        return x        #-- 여기서 리턴 되는 x는 o벡터가 된다.

# 3. 모델 초기화 및 GPU로 이동
model = MLP(input_dim=4, hidden_dim=5, output_dim=3).to(device) #-- .to(device) : 'device' 변수에 들어있는 값(cpu 또는 gpu)에 따라 cpu에서 돌릴지 gpu에서 돌릴지 정해주는 부분
        #-- model : MLP로 만든 인스턴스

# 4. 손실 함수 & Optimizer 설정
criterion = nn.MSELoss() #-- mean square error를 손실함수로 사용하겠다
#-- 손실함수를 다른걸로 정의 할 수도 잇음.
optimizer = optim.SGD(model.parameters(), lr=0.5)
#--  어떤 방식으로 학습 시킬꺼냐 : 경사하강법(sgd)
#-- model.parameters() : 만들어둔 모델의 파라미터들(가중치들)이 리스트 형태로 넘어간다
#-- learning rate는 0.5다.

# 5. Mini-Batch 학습
def train(model, X, Y_onehot, batch_size=10, epochs=1000):
    num_samples = X.size(0)
    loss_history = []
    
    for epoch in range(epochs):     #-- 에포크를 기준으로 루프(여기서 에포크는 50000), [150개의 샘플의 10의 배치 사이즈로 쪼개서 150개의 샘플을 다 본다] : 이게 1 에포크
        # 미니배치 셔플
        indices = torch.randperm(num_samples)       #-- num_sample = 150, 랜덤permutation -> 랜덤 순서 생성
        X = X[indices]      
        Y_onehot = Y_onehot[indices]
        #-- x와 y를 같은 순서로 섞어준다. -> xy 짝이 맞지 않으면 그 데이터셋은 의미가 없잖아
        #-- 원본 데이터를 유지 하려면 
        #-- X1 = X[indices]      
        #-- Y1_onehot = Y_onehot[indices]
        #-- 위 두줄 같은 형태로 하면 원본 데이터 순서대로 존재하는 xy를 유지 가능

        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y_onehot[i:i+batch_size]
            #-- 배치단위로 쪼개줌, 10으로 한다면 [0~9]로 만들어지고 밑의 순전파 역전파 계산되고
            #-- 루프 끝나면 for i in range(0, num_samples, batch_size): 에서 i가 10이 되고 또 그 배치 단위가 순전파 역전파 계산을 수행하는 구조

            # 순전파
            outputs = model(X_batch) #-- 순전파 계산
            loss = criterion(outputs, Y_batch) #-- 출력값과 부류값을 가지고 오류값을 계산

            # 역전파
            optimizer.zero_grad() #-- 파라미터의 미분값을 저장하기 위해서 초기화
            loss.backward() #-- loss값에 대해서 오류역전파 계산 -> 결과 = ??  
            optimizer.step() #-- 파라미터값을 갱신

        if (epoch + 1) % 10 == 0:
            loss_history.append(loss.item())#-- 5000 에포크 중 10 에포크(그러면 10 에포크는 150개의 샘플을 10바퀴 돌고난게 10에포크) 마다 loss 값을 리스트에 저장하고 있음 
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")
    
    return loss_history #--  모든 에포크(5000바퀴)가 끝나면 히스토리 리스트를 반환

#-- 에포크와 배치크기를 헷갈리지 말것. 
#-- 150개의 샘플을 10개로 잘라서 150개의 샘플을 다 보고 난게 1에포크 즉 1바퀴인거고 
#-- 5000에포크를 돌겠다는건 150개의 데이터를 5000바퀴 돌겟다는 것


#-- 학습 시작 시간 측정
start_time = time.time()

loss_history = train(model, X, Y_onehot, batch_size=20, epochs=5000) #-- 히스토리 출력 , 배치사이즈는 20, 에포크는 5000으로 하겠다.

#-- 학습 후 시간 측정
end_time = time.time()

#학습 소요시간 계산 및 출력
elapsed_time = end_time - start_time
print(f"소요 시간: {elapsed_time:.4f}초")

# 6. 예측 및 정확도 평가
with torch.no_grad(): 
    #-- torch.no_grad() 의 역할 = 학습을 위한 계산을 하기 위해 '계산 그래프'를 다 만들어서 각 파라미터의 미분값들을 계산햇음
    #-- 이제 계산이 더 필요없음 
    outputs = model(X) #-- 테스트 셋을 위한 output
    preds = torch.argmax(outputs, dim=1) 
    #-- argmax = 출력값의 최대값을 추출해줌 : (1.23, 0.1, 0.1)-> 첫번째 값을 추출 
    #-- 최종적으로 preds값은 레이블 값이 된다.
    accuracy = (preds == Y).float().mean().item() #-- 논리연산 결과인 0,1 값 /앞에껄 실수화/ 또 그걸 평균구하기/ .item() 하면 정확도값 딱 나온다!
    print(f"\nClassification Accuracy: {accuracy:.4f}")
#-- 보통 트레이닝 셋, 테스트셋을 분리하는데 여기는 샘플수가 적어서 테스트 셋도 150개 그냥 다 한다

# 틀린 샘플 출력
incorrect = (preds != Y).nonzero(as_tuple=False).squeeze()
print("Incorrect predictions:")

if incorrect.shape != torch.Size([]):
   for idx in incorrect:
        print(f"Index: {idx.item()}, Sample: {X[idx].to('cpu').numpy()}, Pred: {preds[idx].item()}, True: {Y[idx].item()}")

# 7. 손실 그래프 (CPU로 이동 필요 없음)
plt.plot(loss_history)
plt.xlabel('Epoch (x10)')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)
plt.show()

#-- gpu에서 해보려면
# 1. 텐서구조로 넣어주고 device 변수에 'cuda'가 들어가도록 해준다. --> 입력 샘플을 cpu 램이 아니라 gpu의 램에 넣어줘야함
X = torch.tensor(X_np, dtype=torch.float32,device=device)
model = MLP(input_dim=4, hidden_dim=5, output_dim=3).to(device) #-- .to(device) : 'device' 변수에 들어있는 값(cpu 또는 gpu)에 따라 cpu에서 돌릴지 gpu에서 돌릴지 정해주는 부분
#-- 입력데이터가 크지 않은 이상 gpu 사용하면 거기로 복사하는 시간이 더 오래 걸릴수 잇음
# 