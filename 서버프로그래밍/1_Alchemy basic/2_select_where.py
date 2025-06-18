import pymysql

# MySQL Connection 연결
conn = pymysql.connect(host='localhost', user='root', password='root',
                       db='madang', charset='utf8')

# Connection 으로부터 Dictoionary Cursor 생성 
# pymysql.cursors.DictCursor는 curs가 dictionary based cursor라는 의미임
curs = conn.cursor(pymysql.cursors.DictCursor)
#-- cursor를 딕셔너리 타입으로 만들주기

# SQL문 실행, %s : 문자열이든 숫자이든 %s 사용
sql = "select * from book where publisher=%s and price>=%s" #-- %s : string의 약자, 보낼때는 string으로 보내야함.
curs.execute(sql, ('이상미디어', 13000))

# 데이타 Fetch
rows = curs.fetchall()
for row in rows:
    print(row)
    # 출력 : {'bookid': 7, 'bookname': '야구의 추억', 'publisher': '이상미디어', 'price': 20000}
    print(row['bookid'], row['bookname'], row['publisher'], row['price'])
    # 7 야구의 추억 이상미디어 20000

# Connection 닫기
conn.close()