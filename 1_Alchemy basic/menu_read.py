import pymysql

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database_setup import Restaurant, Base, MenuItem

engine = create_engine('mysql+pymysql://root:root@localhost/restaurant')
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
# A DBSession() instance establishes all conversations with the database
# and represents a "staging zone" for all the objects loaded into the
# database session object. Any change made against the objects in the
# session won't be persisted into the database until you call
# session.commit(). If you're not happy about the changes, you can
# revert all of them back to the last commit by calling
# session.rollback()
session = DBSession()

firstResult= session.query(Restaurant).first()
print ("first restaurant = "+firstResult.name)
print ("")

# python lotsofmenus.py 실행후에,
restaurants = session.query(Restaurant).all()  # 실행하면, 많은 restaurant이 추가된 것을 확인 가능
for restaurant in restaurants: ## 실행하면 restaurant들이 보여짐
    print (restaurant.name)
print ("")

#-- select all items 와 같은 효과
items = session.query(MenuItem).all()
for item in items: ## 실행하면 item들이 보여짐
    print (item.name)
