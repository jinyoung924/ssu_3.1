from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Restaurant, MenuItem

app = Flask(__name__)

engine = create_engine('mysql+pymysql://root:root@localhost/restaurant')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

# url : http://localhost/restaurants/1/
@app.route('/restaurants/<int:restaurant_id>/')
#-- restaurant_id 변수에 값이 1이 들와서 <int:restaurant_id> 해주면 정수값으로 들어간다.
def restaurantMenu(restaurant_id): #-- project3과 다른 부분은 인자로 restaurant_id를 받는다는 점
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
    return output #-- html 문서 이렇게 만드는거 귀찮음.

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
