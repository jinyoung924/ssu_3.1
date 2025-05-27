import pymysql

# MySQL Connection 연결
conn = pymysql.connect(host='localhost', user='root', password='root',
                       db='madang', charset='utf8')

# Connection 으로부터 Cursor 생성
# curs : array based cursor (default)
curs = conn.cursor()

# restaurant table create 실행
restaurant_sql = '''
CREATE TABLE restaurant 
(id int NOT NULL AUTO_INCREMENT, 
 name varchar(250) NOT NULL,
 PRIMARY KEY (id))
'''
menuitem_sql = '''
CREATE TABLE menu_item 
(id int NOT NULL AUTO_INCREMENT, 
 name varchar(250), 
 price varchar(250),
 description varchar(250) NOT NULL, 
 restaurant_id INTEGER NOT NULL,
 PRIMARY KEY (id),
 FOREIGN KEY(restaurant_id) REFERENCES restaurant(id))
'''

# restaurant, menu_item table 생성 실행
curs.execute(restaurant_sql)
curs.execute(menuitem_sql)

# commit() : table 생성명령어가 실제로 mySQL db에 반영 수행
# curs.commit()

# Connection 닫기
conn.close()