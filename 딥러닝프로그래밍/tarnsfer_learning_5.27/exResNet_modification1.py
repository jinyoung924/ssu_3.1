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


model = resnet18(pretrained=True) 
#-- pretrained = True : 사전 학습된 모델로 resnet18을 model 객체로 생성, (false 면 랜덤한 가중치가 들어간 모델 객체를 생성함)


##### < fc층의 부류값을 10으로 바꾸기 >

num_output = 10
num_ftrs = model.fc.in_features             #-- model의 fc층에 입력으로 들어가는 일차원 벡터의 크기를 저 변수에 저장 (num_ftrs = 512)
model.fc = nn.Linear(num_ftrs,num_output)   
#-- 그리고 우리가 원하는 부류값 num-output(10)으로 바꾸기 위해 
#-- 출력층인 fc층을 
#-- [512크기의 열벡터를 입력받아서 1000개의 부류값을 출력하는 층]에서 
#-- [512크기의 열벡터를 입력받아서 10의 부류값을 출력하는 층]으로 '교체'한다. 


print(model)
model.to(device)
summary(model,input_size=(3,224,224))
# 모델의 정보 요약 출력

#####

transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])
#-- 1. [리사이즈] : cifar-10의 이미지 크기 = (32 * 32) --> resnet의 입력 사이즈인 (224 *224)로 리사이즈
#-- 2. [데이터 증강] : 크롭(자르기), 플립(좌우반전)


training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
#-- 학습데이터와 test데이터를 가져오는 데이터로더 정의


params_name_to_update = ['fc.weight']
params_to_update = []

for name, param in model.named_parameters(): #-- 모든 파라미터를 다 가져온다.
    if 'fc' in name:                         #-- fc에 해당하는 파라미터는 grad 계산을 통해 학습을 시키고
        param.requires_grad = True
        params_to_update.append(param)
    else:                                    #-- fc말고 다른 층에 해당하는 파라미터들은 학습을 시키지 않고 그대로 사용할거다(freezing)
        param.requires_grad = False
#-- 사전 학습 모델의 parameter freezing : fc층만 새로 학습시키려한다.



lr = 1e-4
    #optim = Adam(model.parameters(), lr=lr) : 모든 파라미터를 전부 학습한다
    #optim = Adam(params=params_to_update, lr=lr) : params_to_update에 들어가 있는 파라미터만 학습한다
    #optim = Adam(model.fc.parameters(), lr=lr) : fc층의 파라미터만 학습한다
optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) #-- 이 문장: 옵티마이저가 학습 하려하는 파라미터(fc)만 업데이트하도록 한다.
for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) #-- 에폭마다 학습 진행률 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) #-- 순방향 계산을 통해 모델의 예측값(preds) 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device)) #-- 다중클래스 분류용 손실함수인 크로스엔트로피 함수 사용 
       loss.backward() #-- 역전파 계산
       optim.step() #-- 가중치 업데이트
    
       iterator.set_description(f"epoch:{epoch+1:05d} loss:{loss.item():05.2f}") #-- tqdm이 출력할 문자열
#-- 학습을 돌리는 부분



torch.save(model.state_dict(), "CIFAR_pretrained_ResNet.pth")
model.load_state_dict(torch.load("CIFAR_pretrained_ResNet.pth", map_location=device))
#-- 1. 학습된 모델 저장을 위해 파라미터 파일("CIFAR_pretrained_ResNet.pth": dictionary 형태로 저장)을 만들어서 저장하고
#-- 2. 그 파일에서 파라미터값을 가져와서 model에 다시 로드


num_corr = 0
with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       _, preds = output.data.max(1)
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")
#-- 테스트셋에서 예측 정확도 평가