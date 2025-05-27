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

spinach = session.query(MenuItem).filter_by(name = 'Spinach Ice Cream').one() #-- 스피너치 아이스크림을 하나 가져오고
session.delete(spinach) #--  걔를 삭제 : delete from mnu_item where name = 'Spinach Ice Cream' 와 같은 일을 해줌
session.commit()