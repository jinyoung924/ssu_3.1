from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Restaurant, MenuItem

app = Flask(__name__)

engine = create_engine('mysql+pymysql://root:root@localhost/restaurant')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()


@app.route('/')
@app.route('/restaurants/<int:restaurant_id>/')
def restaurantMenu(restaurant_id):
    restaurant = session.query(Restaurant).filter_by(id=restaurant_id).one()
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id)
    output = ''
    for i in items:
        output += i.name
        output += '</br>'
        output += i.price
        output += '</br>'
        output += i.description
        output += '</br>'
        output += '</br>'
    return output

# Task 1: Create route for newMenuItem function here


@app.route('/restaurant/<int:restaurant_id>/new/') #-- 밑에 def이 수행될 웹페이지 지정 = 라우팅
def newMenuItem(restaurant_id):
    return "page to create a new menu item. Task 1 complete!"


# Task 2: Create route for editMenuItem function here

#-- http://localhost:5000/restaurant/1/2/edit/ -- 이 url의 의미 1번 레스토랑의 2번 메뉴를 edit할 거다.
@app.route('/restaurant/<int:restaurant_id>/<int:menu_id>/edit/') 
#-- 각 웹페이지에서 원하는 작업을 할때 필요한 인자를 함수에 넣고 
#-- 그 인자를 url에 포함시켜서 인자에 맞는 작업은 각 인자가 포함된 url페이지에서 작업하도록 설계
def editMenuItem(restaurant_id, menu_id):
    return "page to edit a menu item. Task 2 complete!" #-- 일단 문자열로 이 페이지에서 수행되어야하는 내용을 적어준것.


# Task 3: Create a route for deleteMenuItem function here

@app.route('/restaurant/<int:restaurant_id>/<int:menu_id>/delete/')
def deleteMenuItem(restaurant_id, menu_id):
    return "page to delete a menu item. Task 3 complete!"


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
