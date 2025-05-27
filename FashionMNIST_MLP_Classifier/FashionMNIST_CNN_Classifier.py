#pytorch를 이용한 간단한 Fashion-MNIST Datatset classifier 구현 
#1. 데이터 작업하기
#(1) 파이토치(PyTorch)에는 데이터 작업을 위한 기본 요소 두가지인 
# torch.utils.data.DataLoader 와 torch.utils.data.Dataset 가 있습니다. 
# Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset을 
# 순회 가능한 객체(iterable)로 감쌉니다.

import torch
from torch import nn
from torch.utils.data import DataLoader         #-- 배치 단위로 dataset을 입력층으로 넣어줌
from torchvision import datasets                #-- dataset을 손쉽게 다룰 수 있도록 해줌
from torchvision.transforms import ToTensor     #-- 텐서로 쉽게 바꾸도록
from torchsummary import summary                #-- 네트워크를 쉽게 분석하게 도와줌

#(2) PyTorch는 TorchText, TorchVision 및 TorchAudio 와 같이 도메인 특화 라이브러리를
# 데이터셋과 함께 제공하고 있습니다. 이 튜토리얼에서는 TorchVision 데이터셋을 사용하도록
# 하겠습니다. Torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 영상(vision)
# 데이터에 대한 Dataset를 포함하고 있습니다. 이 튜토리얼에서는 
# FasionMNIST 데이터셋을 사용합니다. 모든 TorchVision Dataset 은 샘플과 정답을 각각 
# 변경하기 위한 transform 과 target_transform 의 두 인자를 포함합니다.

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(          #-- 옷, 신발, 가방 등의 이미지를 제공하고 분류하는 dataset (28*28, 흑백 (채널이 1개), 전체 7만, 훈련용 6만, test용 1만)
    root="data",
    train=True,             #-- 6만개(28*28의 흑백 이미지 + 10종류중에 하나인 부류값, 레이블값)의 훈련데이터셋을 가져옴
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,            #-- 1만장의 teat 데이터셋을 가져온다.
    download=True,
    transform=ToTensor(),   #-- numpy 데이터를 텐서로 저장
)


#(3)Dataset 을 DataLoader 의 인자로 전달합니다. 
# 이는 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch),
# 샘플링(sampling), 섞기(shuffle) 및 다중 프로세스로 데이터 불러오기(multiprocess data loading)를
# 지원합니다. 여기서는 배치 크기(batch size)를 64로 정의합니다. 즉, 데이터로더(dataloader) 객체의 
# 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환합니다.

batch_size = 64             #-- 배치사이즈를 2의 자승 형태로 설정, gpu의 ram 사이즈에 따라 결정

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)     
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#2. 모델 만들기
#(1) PyTorch에서 신경망 모델은 nn.Module 을 상속받는 클래스(class)를 생성하여 정의합니다.
# __init__ 함수에서 신경망의 계층(layer)들을 정의하고 forward 함수에서 신경망에 데이터를
# 어떻게 전달할지 지정합니다. 가능한 경우 GPU 또는 MPS로 신경망을 이동시켜
# 연산을 가속(accelerate)합니다.

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Convolutional Neural Network 모델을 정의합니다.
class CnnNetwork(nn.Module):            #-- nn.module에서 오버라이드 해서 사용한다.
    def __init__(self):
        super().__init__()              #-- nn.module(부모클래스)의 속성을 초기화 시켜서 사용하겠다.
                                        #-- init 에서는 사용할 네트워크의 컴포넌트를 선언한다.
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)      
         #-- (흑백 사진이니 인풋 채널은 1, 아웃풋 채널이 256이려면 이 층에서 커널을 256개 사용하겠다, 커널사이즈가 3이고 패딩이 1이면 크기변화는 없음, default는 stribe=1 이어서 크기변화 없음)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)      #-- 커널의 수는 64개, 입력과 출력의 사진크기는 변화 없음
        self.linear1 = nn.Linear(in_features=64*7*7, out_features=256)                          #-- 은닉층 : 아웃풋이 256이니깐 출력의 노드수는 256개
        self.linear2 = nn.Linear(in_features=256,out_features=10)                               #-- 출력층 : 입력을 256으로 받아서 부류값의 종류수인 10의 크기로 출력 벡터를 출력함
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                       #-- 풀링층 : 크기를 반으로 줄이겠다, 패딩필요없다!
        self.softmax = nn.Softmax(dim=1)                                      #-- softmax가 출력층의 가장 이상적인 활성함수다.
        self.flatten = nn.Flatten()                                                             #-- flatten : 3차원 입력층을 열벡터로 바꿔줌

    def forward(self, x):       #-- init 에서는 사용할 네트워크의 컴포넌트를 선언하고, forward에서는 어떤 순서로 위에서 선언한 컴포넌트를 이용할건지 정의한다.
        
        #Building Block 1
        x = self.conv1(x) # in_img_size=(28,28), in_channels=1, 
                          # out_channels=256, kernel_size=3, padding=1, out_img_size=(28,28)
        x = self.relu(x)  # in_img_size=(28,28), out_channels=256, out_img_size=(28,28)
        x = self.pool(x)  # in_img_size=(28,28), in_channels=256, kernel_size=2, stride=2
                          # out_channels=256,out_img_size=(14,14)
        
        #Building Block 2 
        x = self.conv2(x) # in_img=(14,14), in_channels=256, out_channels=64, kernel_size=3, stride=1
                          # out_img_size=(14,14), out_channels=64
        x = self.relu(x) # out_img_size=(14,14), out_channels=64
        x = self.pool(x) # in_img_size=(14,14), out_channels=64, kernel_size=2, stride=2
                          # out_img_size=(7,7), out_channels=64
                           
        #Serialization for 2D image * channels                           
        x = self.flatten(x) # in_img_size=(7,7), in_channels=64
                            # out_img_size=(3136,)                          #-- 입려층:7*7*64, 얘의 원소을 일렬로 쫙 세운다 -->3136 크기의 열벡터가 된다.
                            
        #Fully connected layers
        x = self.linear1(x) #in_features=3136, out_features=256
        x = self.relu(x) #in_features=256, out_features=256
        
        #output layer
        x = self.linear2(x) #in_features=256, out_features=10
        #x = self.softmax(x) #in_features=10, out_features=10
        return x

model = CnnNetwork().to(device)
print(model)

summary(model,input_size=(1,28,28))     #-- summary 클래스에게 입풋사이즈를 알려준다. 
                                        #-- terminal 출력에서 표로 보여주는데 각 층에서 사용한 파라미터 수, 각층에서의 출력의 사이즈까지 보여준다.




#-- 위에서 네트워크를 정의 했고 이제 밑에서는 그 네트워크의 가중치를 학습시킬거다





#3. 모델 매개변수 최적화하기
#(1)모델을 학습하려면 손실 함수(loss function) 와 옵티마이저(optimizer)가 필요합니다.

loss_fn = nn.CrossEntropyLoss()                               #-- loss_fn = nn.MSELoss() 얘가 mse보다 성능이 좋다!

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)      #-- optimizer = 경사하강법 (스토캐스틱 그라디언트 디센트)
                                                              #-- model이 위에서 정의한 네트워크인데, parameters() 함수를 이용해서 저 네트워크에 쓰인 파라미터수를 계산해서 인자로 넘겨줌, lr : learning rate


#(2)각 학습 단계(training loop)에서 모델은 (배치(batch)로 제공되는) 학습 데이터셋에 
#대한 예측을 수행하고, 예측 오류를 역전파하여 모델의 매개변수를 조정합니다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):     #--enumerate : 배치 단위로 batch에는 몇번째 배치인지, (X,y)에는 배치 수 만큼 입력샘플과 부류값이 들어간다.
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)                     #-- 순방향 전파로 나온 출력
        loss = loss_fn(pred, y)             #-- 위에서 나온 출력과 부류값을 같이 넣어서 오류값을 계산
        #-- MSE cost function를 사용하려면 윗 두줄을 밑 두줄로 변경해줘야함 
        #-- 왜냐하면 : 크로스엔트로피 함수는 스칼라 y를 내부적으로 원핫인코딩(크기가 10인 열벡터)으로 바꿔서 loss값을 계산해주지만 MSE는 그게 없어서 원핫인코딩으로 바꿔줘야한다.
        #y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
        #loss = loss_fn(pred, y_onehot)
        
        # 역전파
        optimizer.zero_grad()           #-- 그라디언트를 0으로 초기화 한 후
        loss.backward()                 #-- 오류 역전파로 파라미터들의 그라디언트를 계산
        optimizer.step()                #-- step 함수 : 파라미터값의 갱신 ( 그라디언트에 learning rate를 곱한후 빼서 가중치를 업데이트하던 그 계산을 해줌)

        #-- 배치 사이즈 만큼의 입력층을 순방향, 역전파 한 바퀴를 했다 --> 1배치
        #-- 한 에포크를 돌았다 = 모든 입력데이터셋(6만개의 세트)을 배치 단위로 쪼개서 6만개 전체를 다 학습을 시켰다. --> 저 149번째 줄의 for문이 '6만/배치크기'번 돌면 1 epoch가 완료된거다.


        if batch % 100 == 0:        #-- 100번의 배치 마다 중간 결과를 출력한다.(터미널로 확인하는용)
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          

#(3)모델이 학습하고 있는지를 확인하기 위해 테스트 데이터셋으로 모델의 성능을 확인합니다.

#-- 훈련/검증/테스트 data set을 가지고 학습시킴
# validation data set : 훈련에 쓰이지 않은 data set을 검증하는데에 사용. ex)3천개 정도
# 오버피팅문제 : 학습할 파라미터보다 훈련 데이터set이 더 작을 경우에 발생  --> 오버피팅이 발생하는지 검증 데이터셋으로 추적

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # inference 모드로 실행하기 위해 학습시에 필요한 Drouout, batchnorm등의 기능을 비활성화함
    model.eval()                        #-- drop out, batch normalization 은 파라미터 학습에 필요한 연산!
                                        #-- test는 학습이 아니라 forward 프로세싱 만 해서 검사하기 위한 거니깐 .eval() 함수로 위같은 계산을 꺼줘야함.
    test_loss, correct = 0, 0
    with torch.no_grad():               #-- autograd engine(gradinet를 계산해주는 context)을 비활성화함 --> forward 프로세스만 진행할거니깐 (메모리더 적게 들고 시간도 적게 걸린다)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #MSE cost function를 사용할 시 변경해야함
            #y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
            #test_loss += loss_fn(pred, y_onehot).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches        #-- loss값
    correct /= size                 #-- 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") #-- 테스트에 참여한 애들만 loss를 계산
    

#학습 단계는 여러번의 반복 단계 (에폭(epochs)) 를 거쳐서 수행됩니다. 각 에폭에서는 
#모델은 더 나은 예측을 하기 위해 매개변수를 학습합니다. 각 에폭마다 모델의 정확도(accuracy)와 
# 손실(loss)을 출력합니다. 에폭마다 정확도가 증가하고 손실이 감소하는 것을 보려고 합니다.

epochs = 1      #-- 치적 에포크 수는 실험적으로 알아낸다.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")


#4. 모델 저장하기 
# #-- : 저장을 안하고 끄면 파라미터 값이 초기화 되어서 또 다시 학습시켜야하니깐 지금까지 학습된 파라미터 값들을 저장해주는 기능

#모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여)
#내부 상태 사전(internal state dictionary)을 직렬화(serialize)하는 것입니다.

torch.save(model.state_dict(), "CNN_model_for_FASHION_MNIST.pth")
print("Saved PyTorch Model State to model.pth")

#5. Inference 
#이제 이 모델을 사용해서 예측을 할 수 있습니다.

classes = [     #-- 10개의 부류값
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

#-- 최종으로 성능을 확인하기 위한 테스트


###### 중요중요중요 ######
# test에서는 
# 1.model.eval()
# 2.with torch.no_grad(): --> 이 두개의 함수로 파라미터 학습이 아닌 forward processing만 수행하게 한다.

#-- load_state_dict이 저장해둔 학습된 파라미터값 dict이어서 그걸 바로 불러와서 그 값으로 세팅해주기 가능
#model.load_state_dict(torch.load("CNN_model_for_FASHION_MNIST.pth", map_location=device)) 
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

"""
"""
