# 1. 사용자로 부터 comma seprated 입력 문자열을 받아서, list와 tuple을 만들어 출력하시오

input_str = input("Input some comma seprated numbers : ")
list_items = input_str.split(',')
tuple_items = tuple(list_items)
print("List : ", list_items)
print("Tuple : ", tuple_items)


# 2. 사용자로부터 input 숫자 n을 입력받아 n+nn+nnn을 계산하여 출력하시오

n = input("Input an integer : ")
result = int(n) + int(n*10 + n) + int(n*100 + n*10 + n)
print(result)


# 3. 아래 numbers list에서 짝수인 원소만 출력하시오.
#    단, 원소의 값이 237이면 list의 다음 원소값을 읽지 말고 출력을 멈추시오(for-loop에서 break)

numbers = [
    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345,
    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,
    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,
    958,743, 527
]

for num in numbers:
    if num == 237:
        break
    if num % 2 == 0:
        print(num)


# 4. "google.com" string에서 각 character의 숫자를 dictionary로 변환하여 출력하도록
#     char_frequency 함수를 코딩하시오.

def char_frequency(str1):
    freq = {}
    for char in str1:
        freq[char] = freq.get(char, 0) + 1 #키값이 char인 딕셔너리 벨류값을 1증가 시켜준다.
    return freq

print(char_frequency('google.com'))


# 5. 다음 list에서 중복된 원소를 제거한 list를 생성후 출력하시오

my_list = [10, 20, 30, 40, 20, 50, 60, 40]
unique_list = list(set(my_list)) # set()을 이용하면 집합이 된다 -> 자동으로 중복 제거.
print("List of unique numbers : ", unique_list)


# 6. 다음 input dictionary 변수들을 합쳐서 새로운 dictionary로 만든 후 출력하시오

dic1 = {1: 10, 2: 20}
dic2 = {3: 30, 4: 40}
dic3 = {5: 50, 6: 60}

merged_dic = {**dic1, **dic2, **dic3}
print(merged_dic)
# Console Output
# ---------------------------------------------
# {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}


# 7. Input값 숫자 n을 입력하면, key값을 1부터 n까지 하고,
#    각 key에 대한 value를 key*key로 하는 dictionary를 만드시오.

n = int(input("Input an integer : "))
squared_dict = {x: x * x for x in range(1, n + 1)}
print("output :", squared_dict)
# 예시 입력: 10
# Console Output
# ---------------------------------------------
# Input an integer : 10
# output : {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}


# 8. 아래 tuple을 string으로 변환하여 출력하시오

tup = ('e', 'x', 'e', 'r', 'c', 'i', 's', 'e', 's')
joined_string = ''.join(tup)
print("output :", joined_string)
# Console Output
# ---------------------------------------------
# output : exercises


# 9. 아래 2개의 set setx, sety를 합친 set을 구하고 출력하시오

setx = set(["green", "blue"])
sety = set(["blue", "yellow"])
union_set = setx.union(sety)
print(union_set)
# Console Output
# ---------------------------------------------
# {'yellow', 'green', 'blue'}


# 10. List input_list의 각 tuple의 마지막 값을 기준으로 오름차순 정렬하는 함수 만들기

input_list = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]

def sort_list_last(tuples):
    return sorted(tuples, key=lambda x: x[-1])

print(sort_list_last(input_list))
# Console Output
# ---------------------------------------------
# [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)]