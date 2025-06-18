import pymysql

# MySQL Connection 연결
conn = pymysql.connect(host='localhost', user='root', password='root',
                       db='madang', charset='utf8')

# Connection 으로부터 Dictoionary Cursor 생성
# pymysql.cursors.DictCursor는 curs가 dictionary based cursor라는 의미임
curs = conn.cursor(pymysql.cursors.DictCursor)

# SQL문 실행, %s : 문자열이든 숫자이든 %s 사용
sql = "DELETE FROM customer WHERE name = '홍길동'"
curs.execute(sql)

# 데이타 commit
conn.commit();

# Connection 닫기
conn.close()