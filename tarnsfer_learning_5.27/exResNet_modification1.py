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


#-- pretrained = True : 사전 학습된 모델 준비, (false 면 랜덤한 가중치가 들어간 모델 객체를 생성함)
model = resnet18(pretrained=True) # ❶ resnet18 모델 객체 생성 
num_output = 10
num_ftrs = model.fc.in_features             #-- model의 fc층에 입력으로 들어가는 일차원 벡터의 크기를 저 변수에 저장 (num_ftrs = 512)
model.fc = nn.Linear(num_ftrs,num_output)   
#-- 그리고 우리가 원하는 부류값(10)으로 바꾸기 위해 마지막 부분인 fc층을 512크기의 열벡터를 입력받아서 1000개의 부류값을 출력하는 층에서 
#-- 512크기의 열벡터를 입력받아서 10의 부류값을 출력하는 층으로 '교체'한다. 
print(model)

model.to(device)
# 모델의 정보 요약 출력
summary(model,input_size=(3,224,224))

# 데이터 전처리와 증강
transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# 데이터로더 정의
training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#사전 학습 모델의 parameter freezing
params_name_to_update = ['fc.weight']
params_to_update = []

for name, param in model.named_parameters(): #-- 모든 파라미터를 다 가져온다.
    if 'fc' in name:                         #-- fc에 해당하는 파라미터는 grad 계산을 통해 학습을 시키고
        param.requirs_grad = True
        params_to_update.append(param)
    else:                                    #-- fc말고 다른 층에 해당하는 파라미터들은 학습을 시키지 않고 그대로 사용할거다(freezing)
        param.requires_grad = False



# 학습 루프 정의
lr = 1e-4
#optim = Adam(model.parameters(), lr=lr)
#optim = Adam(params=params_to_update, lr=lr)
#optim = Adam(model.fc.parameters(), lr=lr)
optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
    
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1:05d} loss:{loss.item():05.2f}")

torch.save(model.state_dict(), "CIFAR_pretrained_ResNet.pth") # 모델 저장


model.load_state_dict(torch.load("CIFAR_pretrained_ResNet.pth", map_location=device))
num_corr = 0
with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       _, preds = output.data.max(1)
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")