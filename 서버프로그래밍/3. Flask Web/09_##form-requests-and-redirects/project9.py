from flask import Flask, render_template, request, redirect, url_for
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
    if restaurant_id == None:
        restaurant_id = 1
    restaurant = session.query(Restaurant).filter_by(id=restaurant_id).one()
    items = session.query(MenuItem).filter_by(restaurant_id=restaurant.id)
    return render_template('menu.html', restaurant=restaurant, items=items)

# Task 1: Create route for newMenuItem function here




@app.route('/restaurant/<int:restaurant_id>/new/', methods=['GET', 'POST']) #-- 메소드로 get,post 넣음, 지정 안하면 default로 get만 지원
# 실행 순서 : 1) else문의 newmenuitem.html 실행,
#            2) create button click하면 form tag의 action에 있는 url 실행 (if문 실행)
#            3) if문의 마지막 함수인 redirect를 부르면 restaurantMenu 함수를 실행
def newMenuItem(restaurant_id):
    if request.method == 'POST': #-- request는 flask에 내장된 객체
        #-- 2. request중에 post인 애는 밑을 수행
        newItem = MenuItem( name=request.form['name'], restaurant_id=restaurant_id)
        #-- 3. request의 form에서 입력받은 정보를 읽어온다. 
        #       request.form['name'] : form tag의 name에 들어가 있는 string
        #       MenuItem(name="불고기버거", restaurant_id=2)    * MenuItem은 sqlAlchemy의 문법임
        #       : 이건 sqlAlchemy에서 불고기버거를 ID가 2번인 식당의 메뉴로 등록 하겠다는 뜻. 
        #       --> restaurant_id(추가하려는 데이터의 cloum)가 이 함수의 인자인 식당id(restaurant_id)인 데이터 행에 form에서 입력받은 string을 이름으로 넣겠다.

        session.add(newItem)
        session.commit()
        # 세션 설명 추가!!

        # redirect : 다른 url로 이동
        return redirect(url_for('restaurantMenu', restaurant_id=restaurant_id))
    #-- 4. redirect(url 주소) : url로 라우팅 되어있는 함수( = @app.route(url주소) 밑에 있는 함수)로 브라우저를 이동시킨다.
    #      url_for('보내려는 함수의 이름', 같이 보내려는 인자 = 값)
    #      저 함수의 url을 적어줘야하는데 간편하게 url_for를 이용해서 'restaurantMenu'로 이동한다.

    else:   # request.method == 'GET'
        return render_template('newmenuitem.html', restaurant_id=restaurant_id)




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
