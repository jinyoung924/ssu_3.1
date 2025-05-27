#!/usr/bin/env python3
#
# Client for the UINames.com service.

import requests


def SampleRecord():
    # http://citibikenyc.com/stations/json을 실행하면 json data를 return함.
    r = requests.get("https://jsonplaceholder.typicode.com/users", timeout=2.0)

    # requests.get 메소드에서 return된 r은 JSON format data인데,
    # json()메소드를 실행하면, dictionary data type으로 변환 (j는 dictionary data type)
    j = r.json()
    # 첫번째 station 객체 가져옴
    user = j[0]

    return "Useer name is {}, email address is {} and phone number is {}.".format(
        user["name"],
        user["email"],
        user["phone"]
    )

if __name__ == '__main__':
    print(SampleRecord())
