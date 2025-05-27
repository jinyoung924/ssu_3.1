# insert_data.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup_book import Base, BookStore, BookItem

# DB 연결 문자열 (비밀번호에 ! → %21로 인코딩)
engine = create_engine('mysql+pymysql://root:kjygoo0924%21@localhost/bookstore')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

# 1. BookStore 추가
store = BookStore(name="성진서점")
session.add(store)
session.commit()

# 2. BookItem 추가 (store.id가 1번이라고 가정)
book1 = BookItem(name="혼공파이썬", price="25000", bookstore_id=store.id)
book2 = BookItem(name="Do it! 플라스크 웹 개발", price="29000", bookstore_id=store.id)

session.add_all([book1, book2])
session.commit()

print("데이터 추가 완료")