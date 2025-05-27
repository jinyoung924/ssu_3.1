import pymysql

# MySQL Connection 연결
conn = pymysql.connect(host='localhost', user='root', password='root',
                       db='madang', charset='utf8')

# Connection 으로부터 Cursor 생성
# curs : array based cursor (default)
#-- 쓰려하는 데이터베이스의 범위를 cursor로 잡는다. 그 범위는 array로 작동 
curs = conn.cursor()

# SQL문 실행
sql = "select * from customer"
curs.execute(sql)

# 데이타 Fetch
rows = curs.fetchall()
print(rows)  # 전체 rows
# print(rows[0])  # 첫번째 row: (1, '박지성', '영국 맨체스타', '000-5000-0001')
# print(rows[1])  # 두번째 row: (2, '김연아', '대한민국 서울', '000-6000-0001')
#-- 이 데이터 타입은 튜플임

# Connection 닫기
conn.close()
#-- 다른 클라이언트도 쓸 수 있도록 close 해야됨.