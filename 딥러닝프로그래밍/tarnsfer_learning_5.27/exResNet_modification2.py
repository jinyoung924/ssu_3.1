import tqdm
import torch
import torch.nn as nn
from torchsummary import summary

from torchvision.models.resnet import resnet18

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam


device = "cuda" if torch.cuda.is_available() else "cpu"
#-- gpu 있다면 gpu 써라

model = resnet18(pretrained=True) #-- 사전 학습된 모델인 resnet18을 model 객체 생성
num_output = 10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,num_output)
model.to(device)
print(model)
#-- resnet 18을 객체로 불러오고
#-- fc층을 바꿔서 1000개의 부류값을 출력하던 출력층에서 10개의 부류값을 출력하는 출력층으로 교체

summary(model,input_size=(3,224,224))
#-- model의 모델 구조, 파라미터수를 출력


transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])
#-- 1. 리사이즈, 크롭, 플립 데이터 전처리와 증강
#-- 2. 정규화 : 신경망은 0근처의 값을 가질때 학습이 잘되어서 정규화 해준다.



training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
# 데이터로더 정의

#
params1_name_to_update = [] #-- layer4의 파라미터가 저장되는 곳
params2_name_to_update = [] #-- fc층의 파라미터가 저장되는 곳
#
params1_to_update = []
params2_to_update = []
for name, param in model.named_parameters():

    #
    if 'layer4' in name:                            #-- 여기에서는 fc층 직전에 있는 feature층 (layer4) '하나'만 학습을 시켜보겠다.
        param.requires_grad = True                  #-- 학습 시키겠다 = "param.requirs_grad = True"
        params1_to_update.append(param)
        params1_name_to_update.append(name)
    elif 'fc' in name:                              #-- fc층은 당연히 학습 시키고!
        param.requires_grad = True
        params2_to_update.append(param)
        params2_name_to_update.append(name)
    else:
        param.requires_grad = False                 #-- layer4와 fc층 빼고는 freezing
print(params1_name_to_update)
print(params2_name_to_update)
#-- model의 층의 파라미터들을 선택적으로 학습시키고 그대로 사용(freezing)하고 설정해준다 



lr1 = 1e-6
lr1 = 1e-4
#optim = Adam(model.parameters(), lr=lr)
optim = Adam([
              {'params':params1_to_update, 'lr':1e-6},  #-- 레이어4에 해당하는 파라미터들은 'lr':1e-6 의 러닝레이트로 학습시키고
              {'params':params2_to_update, 'lr':1e-4}]) #-- classifier는 'lr':1e-4 의 러닝레이트로 학습 시키겠다. 
#-- 두 파라미터들을 학습시키는 방법이 다른데 이렇게 하는 이유 : feature 층들은 이미 학습이 완료된 애들이고 classifier층들은 우리가 새로 갈아끼운 층들이라 초기값이 랜덤값인 학습이 되지 않은 층이다.
#--  -> 이미 학습을 거쳐서 최적화에 가까운 features 층들의 파라미터는 낮은 학습률로 학습시키는게 최적화에 더 가까운 값을 만들어 낼 것이고 
#--     classifier 층들의 파라미터는 학습률을 비교적 높게해야 더 효과적이고 학습이 완료되는데에 걸리는 시간도 줄어든다.

for epoch in range(5): #-- 5에폭동안 학습한다
   iterator = tqdm.tqdm(train_loader) #-- tqdm으로 학습 진행률 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
#--  학습이 이뤄지는 곳.


torch.save(model.state_dict(), "CIFAR_pretrained_ResNet.pth") # 모델 저장
model.load_state_dict(torch.load("CIFAR_pretrained_ResNet.pth", map_location=device))
#-- 모델 저장 & 다시 불러오기


num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       _, preds = output.data.max(1)
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")