from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Restaurant, MenuItem
app = Flask(__name__)

#-- 데이터베이스와 서버를 연결해보기 

#-- 세션만들기
engine = create_engine('mysql+pymysql://root:root@localhost/restaurant')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()


@app.route('/')
@app.route('/hello')
def HelloWorld():
    restaurant = session.query(Restaurant).first() #-- 레스토랑의 첫번째 쿼리를 가져옴 (아이디가 1인 urbun bugger)
    # output = restaurant.name
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id) 
    #-- restaurant_id가 위에서 가져온 레스토랑의 restaurant.id인 레스토랑의 메뉴를 가져와라
    output = ''
    for i in items:
        output += i.name
        output += '</br>'
    return output #-- 메뉴 이름을 리턴

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
