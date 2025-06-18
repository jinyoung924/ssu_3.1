import tqdm
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader

from torchvision.models.vgg import vgg16
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torchsummary import summary

from torch.optim.adam import Adam


device = "cuda" if torch.cuda.is_available() else "cpu"
#-- cuda 가능하면 gpu 사용


model = vgg16(pretrained=True) #-- 사전 학습된 vgg16을 model의 객체로 생성
print(model)
model = model.to(device)
summary(model,input_size=(3,224,224))   #-- 모델의 요약 정보를 담는 객체
#-- model의 모델 구조와 파라미터 수 출력


for i,(name,param) in enumerate(model.named_parameters()):  #-- model.named_parameters(): 모델의 모든 파라미터를 이름과 같이 순회하도록 리턴, enumerate(): 반복문 순회를 위한 문법
    print(f"{name}:param.requires_grad-->{param.requires_grad}")
    param.requires_grad = False
#1. 모든 파라미터를 고정 (freezing)
for i,(name,param) in enumerate(model.features.named_parameters()):  
    #-- 이렇게 하면 features층의 파라미터 값을 freezing, classifier층의 파라미터는 최종 부류 값을 1000에서 10으로 바꿀거니깐 수정해서 쓸거다. 그래서 여기는 freezing 안함.
    print(f"{name}:param.requires_grad-->{param.requires_grad}")
    param.requires_grad = False
#2. 모든 feature extractor의 파라미터를 고정(freezing), classifier는 학습이 가능한 상태가 된다.
#-- 1,2 원하는걸 선택해서 freezing 하면 된다.



class fcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512 * 7 * 7,4096)
        self.dropout = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(4096,4096)
        self.linear3 = nn.Linear(4096,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
#-- Fully connected layer 모델을 정의
#-- vgg16의 feature extracter의 출력값인 512 * 7 * 7 크기를 입력으로 받는 classifier(mlp, fc층)을 새롭게 정의    

#fc = nn.Sequential(                    #-- ❷ 분류층의 정의
#       nn.Linear(512 * 7 * 7, 4096),
#       nn.ReLU(),
#       nn.Dropout(),                   #-- ❷ 드롭아웃층 정의
#       nn.Linear(4096, 4096),
#       nn.ReLU(),
#       nn.Dropout(),
#       nn.Linear(4096, 10),
#   )
#-- nn.시퀀셜 을 이용해 직관적으로 작성할 수도 있다.


fc = fcNet()
model.classifier = fc #-- VGG의 classifier를 덮어씀
#-- 기존 분류기 말고 새로 정의해서 사용할 건데 그 분류기의 객체가 fc이다.


print(model)
model.to(device)
#-- model의 모델 구조를 출력해서 확인하는용
#-- model을 device(cpu나 gpu)에 적재해서 계산 하는데에 사용



transforms = Compose([
   ToTensor(),
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])
#-- 데이터 전처리와 증강
#-- 리사이즈, 크롭, 플립, 정규화


training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
#-- 데이터로더 정의 : 데이터를 로드 해올때 dataset을 다운받고 전처리를 적용한다.


loss_fn = nn.CrossEntropyLoss() #-- 크로스엔트로피함수 : 분류 문제를 풀때 사용하는 손실함수
#lr = 1e-4
#optim = Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-2) 
#-- model.parameters() : 모델 전체의 파라미터 값들, 
#-- model.classifier.parameters() : 위에서 새로 정의한 classifier(분류기)의 파라미터
#< optimizer의 종류 >
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) : 전체 파라미터 (feature , classifier에 있는 파라미터 전부)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)       :학습가능한 파라미터만 optimizer에게 넘겨주겠다.
    #-- lamda는 익명의 함수를 즉석에서 만들어주는 파이썬 키워드. 저기에서 정의된 lamda는 grad계산 할 애들만 리턴한다.
#-- 손실함수 정의 + optimizer(파라미터를 업데이트하는 도구)에게 어떤 파라미터만 업데이트할건지 정의해준다
    

for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) # 터미널에서 학습 진행률(로그)을 processing바 형태로 확인가능
   for data, label in iterator:
       optimizer.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = loss_fn(preds, label.to(device))
       loss.backward()
       optimizer.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1:03d} loss:{loss.item():05.3f}")#-- loss값 확인가능
#-- 학습이 이뤄지는 곳
#-- 5 에폭 동안 학습한다. : 1에폭의 의미 = 모든 학습데이터 셋을 이용하여 학습해서 모든 파라미터들은 학습될 기회를 한번 갖는다.
#   -> 모든 학습데이터셋을 5번 이용해서 모든 파라미터들이 각각 5번 업데이트 된다.


torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장
model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))
#-- model의 파라미터 값을 저장하고, 그걸 다시 불러옴


num_corr = 0
with torch.no_grad(): #-- 정확도 측정할때는 메모리 절약을 위해 no.grad() 사용
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")
#-- accuracy 측정
