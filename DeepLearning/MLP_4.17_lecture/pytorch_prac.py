import torch
import numpy as np
t= torch.tensor([1,2,3], device = 'cpu', require_grad = False, dtype=torch.float32)
# ( 텐서에 담을 객체, 텐서를 어디서 계산 할것인지, 미분이 필요한 계산이면 true로, 행렬 원소의 자료형)

t2= t.to(torch.device('cuda')) #t텐서를 t2라는 변수로 gpu에 생성

#행렬생성
arr = [[1,2,], [3,4]]
np.array(arr) #numpy array 객체 생성
torch.Tensor(arr) #파이토치 텐서 객체 생성
torch.ones((2,2)) #1의 값을 같은 2x2행렬(np도 동일)
#난수생성
np.random.seed(0) #이 밑으로 어떤 랜덤 함수에서든 생성되는 난수값의 순서가 같다.
np.random.rand(2,2) #여기선 1~4번째 난수가 쓰이고 3x3에서는 5~13번째 난수가 쓰인다
np.random.seed(0) #seed를 다시 써서 난수 시퀀스를 초기화 해서 다시 1~4번째 난수 출력할 수 있다
torch.manual_seed(0)
#numpy에서 torch로
np_array = np.ones((2,2))
torch_tensor = torch.from_numpy(np_array)
np_array_new = np.ones((2,2), dtype=np.float64)#int:정수, uint8:0~255,float64:소수점 표현이 두배
torch.from_numpy(np_array_new)
#torch에서 numpy로
torch_to_numpy = torch_tensor.numpy()
#다양한 배열이나 텐서
torch.eye(3, dtype=torch.double)#eye:대각행렬, zeros:0행렬,ones:1행렬
torch.arange(1,10,1)#1부터10까지1씩증가,

#파이토치 차원조작
x = torch.randn(2, 3)
# 모양바꾸기
x_reshaped = x.reshape(3, 2)  # 메모리 연속 여부 신경 안 씀, 연속 아니면 복사해서 만들어줌
x_viewed = x.view(3, 2)       # 메모리가 연속이어야 작동함
x_transposed = x.transpose(0, 1)  # shape (3, 2)   : dim=0 차원과 dim=1차원을 바꾸겠다.

x_unsqueezed_0 = x.unsqueeze(0)  # shape (1, 2, 3) : dim=0에 1인 차원을 추가하겠다.
x_unsqueezed_1 = x.unsqueeze(1)  # shape (2, 1, 3)
x_unsqueezed_2 = x.unsqueeze(2)  # shape (2, 3, 1)
x_squeezed = x.unsqueeze(0).squeeze()  # (2, 3) → (1, 2, 3) → (2, 3)

t1 = torch.ones(2, 3)
t2 = torch.zeros(2, 3)

x_cat_dim0 = torch.cat((t1, t2), dim=0)  # shape (4, 3)
x_cat_dim1 = torch.cat((t1, t2), dim=1)  # shape (2, 6)

x_stack_dim0 = torch.stack((t1, t2), dim=0)  # shape (2, 2, 3)
x_stack_dim1 = torch.stack((t1, t2), dim=1)  # shape (2, 2, 3)

#평균 계산
print("=== 1차원 텐서 ===")
t1 = torch.tensor([1, 2, 3, 4, 5])
print("원본 텐서:", t1)
print("전체 평균:", t1.mean())           # 전체 평균
print("dim=0 평균:", t1.mean(dim=0))     # dim=0도 전체 평균 (1D에서는 동일)

print("\n=== 2차원 텐서 ===")
t2 = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])
print("행별 평균 (dim=1):", t2.mean(dim=1))   # 각 행의 평균 → 결과 shape: [2]
print("열별 평균 (dim=0):", t2.mean(dim=0))   # 각 열의 평균 → 결과 shape: [5]


#deep,shallow 복사
print("=== numpy -> torch.as_tensor() (공유) ===")
arr = np.array([1, 2, 3])
t1 = torch.as_tensor(arr)
arr[0] = 100
print("numpy 수정 후:", arr)     # [100, 2, 3]
print("tensor 반영됨:", t1)      # tensor([100, 2, 3])

print("\n=== numpy -> torch.tensor() (복사) ===")
arr2 = np.array([1, 2, 3])
t2 = torch.tensor(arr2)
arr2[0] = 999
print("numpy 수정 후:", arr2)    # [999, 2, 3]
print("tensor 영향 없음:", t2)   # tensor([1, 2, 3])

print("\n=== 텐서 clone() (복사) ===")
t3 = torch.tensor([10, 20, 30])
t3_clone = t3.clone()
t3[0] = 999
print("원본:", t3)                # tensor([999, 20, 30])
print("복사본:", t3_clone)        # tensor([10, 20, 30])

print("\n=== view()는 메모리 공유 ===")
t4 = torch.tensor([[1, 2], [3, 4]])
t4_view = t4.view(4)
t4[0][0] = 999
print("원본 변경 후 view:", t4_view)  # tensor([999, 2, 3, 4])

print("\n=== reshape()는 상황에 따라 다름 ===")
# reshape는 view처럼 작동하지만, 메모리 연속이 아닐 경우 복사할 수도 있음
t5 = torch.tensor([[1, 2], [3, 4]])
t5_reshaped = t5.reshape(4)
t5[0][1] = 888
print("원본 변경 후 reshape:", t5_reshaped)  # 보통 변경됨 → 공유된 경우

print("\n=== type casting은 복사 ===")
t6 = torch.tensor([1.0, 2.0])
t6_int = t6.int()
t6[0] = 9.9
print("원본:", t6)               # tensor([9.9, 2.0])
print("형변환된 복사본:", t6_int) # tensor([1, 2])

#다른유용한 함수
# 1. requires_grad=True로 텐서를 만들면 계산 그래프가 생성됨
t = torch.tensor([2.5, 3.5], requires_grad=True)
print("원본 텐서:", t)
print("requires_grad:", t.requires_grad)

# 2. detach()로 계산 그래프에서 분리된 새 텐서 만들기 (복사 아님)
t_detached = t.detach()
print("\ndetach() 결과:", t_detached)
print("requires_grad:", t_detached.requires_grad)  # False

# 3. t[0]은 여전히 그래프의 일부
print("\nt[0]:", t[0])  # grad_fn 있음
print("t[0]의 grad_fn:", t[0].grad_fn)

# 4. item()으로 값만 파이썬 숫자(float)로 추출
value = t[0].item()
print("\nt[0].item()의 결과:", value)
print("타입:", type(value))  # float

# 5. NumPy로 변환하려면 먼저 detach + cpu() + numpy()
np_array = t.detach().cpu().numpy()
print("\nNumPy 변환 결과:", np_array)
print("NumPy 타입:", type(np_array))  # <class 'numpy.ndarray'>