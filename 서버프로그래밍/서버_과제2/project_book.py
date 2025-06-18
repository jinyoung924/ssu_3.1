# =================================================================
# 1. 라이브러리 및 모듈 임포트
# =================================================================
# Flask: 웹 애플리케이션을 생성하는 핵심 클래스
# render_template: HTML 파일을 렌더링(보여주기)하는 함수
# request: 클라이언트(브라우저)가 보낸 요청 정보를 담고 있는 객체 (e.g., 폼 데이터)
# redirect: 사용자를 다른 URL로 이동시키는 함수
# url_for: URL 주소를 동적으로 생성해주는 함수
# jsonify: 파이썬 딕셔너리를 JSON 응답으로 변환하는 함수
from flask import Flask, render_template, request, redirect, url_for, jsonify

# SQLAlchemy: 데이터베이스 작업을 쉽게 하도록 도와주는 라이브러리(ORM)
# create_engine: 데이터베이스와 연결을 설정하는 함수
from sqlalchemy import create_engine
# sessionmaker: 데이터베이스와 통신(CRUD)하는 세션을 생성하는 클래스
from sqlalchemy.orm import sessionmaker
# Base, BookStore, BookItem: 데이터베이스 테이블과 매핑되는 파이썬 클래스 (별도 파일에 정의)
from database_setup_book import Base, BookStore, BookItem

# =================================================================
# 2. Flask 애플리케이션 및 데이터베이스 설정
# =================================================================

# Flask 애플리케이션 객체를 생성합니다. __name__은 현재 파일의 이름을 의미합니다.
app = Flask(__name__)

# 데이터베이스 연결 엔진을 생성합니다. 'mysql+pymysql'은 DB 종류, 그 뒤는 접속 정보입니다.
# 형식: 'DB종류+드라이버://사용자명:비밀번호@호스트/DB이름'
engine = create_engine('mysql+pymysql://root:kjygoo0924!@localhost/bookstore')

# 데이터베이스 모델(테이블)들이 어떤 엔진을 사용해야 하는지 알려줍니다.
Base.metadata.bind = engine

# 데이터베이스와 대화할 수 있는 세션(DBSession)을 만듭니다.
# 이 세션을 통해 모든 CRUD 작업을 수행합니다.
DBSession = sessionmaker(bind=engine)
session = DBSession()


# =================================================================
# 3. 라우팅(Routing) 및 뷰(View) 함수 구현
# =================================================================
# @app.route()는 특정 URL 주소와 파이썬 함수를 연결하는 '데코레이터'입니다.
# 사용자가 해당 URL로 접속하면 바로 아래에 있는 함수가 실행됩니다.

# API 엔드포인트: 서점의 모든 책 목록을 JSON 형식으로 반환합니다.
# JSON은 프로그램 간 데이터 교환에 주로 사용됩니다.
@app.route('/bookstores/<int:bookstore_id>/booklist/JSON')
def bookListJSON(bookstore_id):
    # 'bookstore_id'에 해당하는 모든 BookItem을 데이터베이스에서 조회합니다.
    items = session.query(BookItem).filter_by(bookstore_id=bookstore_id).all()
    # 조회된 책 목록을 JSON 형식으로 변환하여 반환합니다.
    # .serialize는 database_setup_book.py 파일 내에 정의된, 객체를 딕셔너리로 바꿔주는 메서드입니다.
    return jsonify(BookItems=[item.serialize for item in items])


# 메인 페이지: 특정 서점의 책 목록을 HTML 페이지로 보여줍니다.
# 하나의 함수에 여러 URL을 연결할 수도 있습니다.
@app.route('/')
@app.route('/bookstores/<int:bookstore_id>/booklist')
def bookList(bookstore_id=None):
    # 만약 URL에 bookstore_id가 없다면 기본값으로 1번 서점을 보여줍니다.
    if bookstore_id is None:
        bookstore_id = 1
    # 해당 ID의 서점 정보와 책 목록을 DB에서 조회합니다.
    bookstore = session.query(BookStore).filter_by(id=bookstore_id).one()
    items = session.query(BookItem).filter_by(bookstore_id=bookstore_id).all()
    # 'booklist.html' 템플릿에 조회한 데이터를 전달하여 화면을 생성하고 사용자에게 보여줍니다.
    return render_template('booklist.html', bookstore=bookstore, items=items, bookstore_id=bookstore_id)


# 새 책 추가 기능: GET 방식과 POST 방식을 모두 처리합니다.
# methods=['GET', 'POST']는 이 함수가 두 가지 요청 방식을 모두 처리할 수 있음을 의미합니다.
@app.route('/bookstores/<int:bookstore_id>/new', methods=['GET', 'POST'])
def newBookItem(bookstore_id):
    # request.method를 통해 현재 요청이 GET인지 POST인지 확인합니다.
    if request.method == 'POST':
        # POST 요청일 경우 (사용자가 폼을 작성하고 '제출' 버튼을 눌렀을 때):
        # request.form['name']과 같이 폼에서 전송된 데이터를 가져옵니다.
        newItem = BookItem(
            name=request.form['name'],
            price=request.form['price'],
            bookstore_id=bookstore_id
        )
        # 새로운 책 객체를 세션에 추가(add)하고 데이터베이스에 최종 반영(commit)합니다.
        session.add(newItem)
        session.commit()
        # 책 추가 후에는 해당 서점의 목록 페이지로 사용자를 이동(redirect)시킵니다.
        # url_for('bookList', ...)는 'bookList' 함수의 URL을 동적으로 생성해줍니다.
        return redirect(url_for('bookList', bookstore_id=bookstore_id))
    else:
        # GET 요청일 경우 (사용자가 처음 '새 책 추가' 페이지에 접속했을 때):
        # 'newbook.html' 템플릿을 렌더링하여 사용자에게 입력 폼을 보여줍니다.
        return render_template('newbook.html', bookstore_id=bookstore_id)


# 책 정보 수정 기능
@app.route('/bookstores/<int:bookstore_id>/<int:book_id>/edit', methods=['GET', 'POST'])
def editBookItem(bookstore_id, book_id):
    # 수정할 책의 정보를 DB에서 먼저 조회합니다.
    editedItem = session.query(BookItem).filter_by(id=book_id).one()
    if request.method == 'POST':
        # 사용자가 수정한 내용을 폼에서 받아와 객체의 속성을 변경합니다.
        if request.form['name']:
            editedItem.name = request.form['name']
        if request.form['price']:
            editedItem.price = request.form['price']
        # 변경된 객체를 세션에 추가하고 DB에 반영합니다. (SQLAlchemy는 객체의 변경을 감지하여 UPDATE 쿼리를 실행)
        session.add(editedItem)
        session.commit()
        return redirect(url_for('bookList', bookstore_id=bookstore_id))
    else:
        # GET 요청 시, 기존 책 정보가 채워진 수정 폼을 사용자에게 보여줍니다.
        return render_template(
            'editbook.html', bookstore_id=bookstore_id, book_id=book_id, item=editedItem)


# 책 삭제 기능
@app.route('/bookstores/<int:bookstore_id>/<int:book_id>/delete', methods=['GET', 'POST'])
def deleteBookItem(bookstore_id, book_id):
    # 삭제할 책의 정보를 DB에서 조회합니다.
    itemToDelete = session.query(BookItem).filter_by(id=book_id).one()
    if request.method == 'POST':
        # POST 요청 시 (사용자가 삭제 확인 버튼을 눌렀을 때):
        # 해당 객체를 세션에서 삭제(delete)하고 DB에 반영(commit)합니다.
        session.delete(itemToDelete)
        session.commit()
        return redirect(url_for('bookList', bookstore_id=bookstore_id))
    else:
        # GET 요청 시, 사용자에게 정말로 삭제할 것인지 확인하는 페이지를 보여줍니다.
        return render_template('deletebook.html', item=itemToDelete)

# =================================================================
# 4. 서버 실행
# =================================================================

# 이 파일이 직접 실행될 경우에만 app.run()을 호출합니다.
# 다른 파일에서 이 파일을 모듈로 임포트할 경우에는 서버가 실행되지 않습니다.
if __name__ == '__main__':
    # 디버그 모드를 활성화합니다. 코드가 변경될 때마다 서버가 자동으로 재시작되고,
    # 오류 발생 시 웹 페이지에 자세한 오류 정보가 표시됩니다. (개발 시에만 사용)
    app.debug = True
    # 웹 서버를 실행합니다. host와 port를 지정할 수 있습니다.
    app.run(host='127.0.0.1', port=5001)