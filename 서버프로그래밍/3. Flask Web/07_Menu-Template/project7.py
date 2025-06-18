from flask import Flask, render_template
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
def restaurantMenu(restaurant_id=None):
    if restaurant_id == None:
        restaurant_id = 1
    restaurant = session.query(Restaurant).filter_by(id=restaurant_id).one()
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id)
    # render_template함수는 'menu.html 파일을 찾을 때 default로 templates 디렉토리에서 파일을 검색
    return render_template('menu.html', restaurant=restaurant, items=items) 
#--render_template('인자를 보낼 html', 식당id = 1, items= 식당의 메뉴들) -> 파라미터들: html로 전달됨

#-- jinja2 template 라이브러리 : flask에서 지원하는 html template
#   -> html 안에 코딩을 할 수 있게 해줘서 html 코드를 일일이 쓰는게 아니라 for문을 돌리는것 같이 편하게 만들어주는 역할 



# Task 1: Create route for newMenuItem function here

@app.route('/restaurant/<int:restaurant_id>/new/')
def newMenuItem(restaurant_id):
    return "page to create a new menu item. Task 1 complete!"

# Task 2: Create route for editMenuItem function here

@app.route('/restaurant/<int:restaurant_id>/<int:menu_id>/edit/')
def editMenuItem(restaurant_id, menu_id):
    return "page to edit a menu item. Task 2 complete!"

# Task 3: Create a route for deleteMenuItem function here


@app.route('/restaurant/<int:restaurant_id>/<int:menu_id>/delete/')
def deleteMenuItem(restaurant_id, menu_id):
    return "page to delete a menu item. Task 3 complete!"


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
