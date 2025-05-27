# 사전 학습된 모델 준비

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
model = vgg16(pretrained=True) # ❶ vgg16 모델 객체 생성

print(model)


# 모델의 정보 요약 출력
model = model.to(device)
summary(model,input_size=(3,224,224))   #-- 모델의 요약 정보를 담는 객체


for i,(name,param) in enumerate(model.named_parameters()):  #-- 파라미터마다 이름을 가지고 있다.
    print(f"{name}:param.requires_grad-->{param.requires_grad}")
    param.requires_grad = False

# 모든 feature extractor의 파라미터를 고정 (freezing)
#for param in vgg16.features.parameters():
#    param.requires_grad = False   

for i,(name,param) in enumerate(model.features.named_parameters()):  
    #-- 이렇게 하면 features층의 파라미터 값을 freezing, classifier층의 파라미터는 최종 부류 값을 1000에서 10으로 바꿀거니깐 수정해서 쓸거다. 그래서 여기는 freezing 안함.
    print(f"{name}:param.requires_grad-->{param.requires_grad}")
    param.requires_grad = False

# Fully connected layer 모델을 정의합니다.
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
    
#-- nn.sequrntial 은 classifier , 분류기 역할을 한다. 
#-- 전이 학습에서는 features(특징맵 만드는 층들)은 그대로 두고(freezing), classifier(분류기 역할을 하는 층들)을 고쳐서 사용하겠다.

#fc = nn.Sequential(                    #-- ❷ 분류층의 정의
#       nn.Linear(512 * 7 * 7, 4096),
#       nn.ReLU(),
#       nn.Dropout(),                   #-- ❷ 드롭아웃층 정의
#       nn.Linear(4096, 4096),
#       nn.ReLU(),
#       nn.Dropout(),
#       nn.Linear(4096, 10),
#   )

fc = fcNet()
model.classifier = fc # ➍ VGG의 classifier를 덮어씀
#-- 기존 분류기 말고 새로 정의해서 사용할 건데 그 분류기의 객체가 fc이다.

#for i,(name,param) in enumerate(model.named_parameters()):
#    print(f"{name}:param.requires_grad-->{param.requires_grad}")
#    param.requires_grad = False

print(model)

model.to(device)
# 데이터 전처리와 증강
transforms = Compose([
   ToTensor(),
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# 데이터로더 정의

training_data = CIFAR10(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 학습 루프 정의"

#lr = 1e-4
#optim = Adam(model.parameters(), lr=lr)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-2) 
#-- model.parameters() : 모델 전체의 파라미터 값들, model.classifier.parameters() : 위에서 새로 정의한 classifier(분류기)의 파라미터
#-- 밑은 optimizer의 종류
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) : 전체 파라미터 (feature , classifier에 있는 파라미터 전부)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)       :학습가능한 파라미터만 optimizer에게 넘겨주겠다.
    #-- lamda는 익명의 함수를 즉석에서 만들어주는 파이썬 키워드. 저기에서 정의된 lamda는 grad계산 할 애들만 리턴한다.

    
#-- 터미널에서 학습 진행정도를 processing 바 형태로 확인 가능 + loss값 확인 가능
for epoch in range(5):
   iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
   for data, label in iterator:
       optimizer.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = loss_fn(preds, label.to(device))
       loss.backward()
       optimizer.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1:03d} loss:{loss.item():05.3f}")



torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장

model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")

"""   
"""