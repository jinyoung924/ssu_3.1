from flask import Flask
app = Flask(__name__)


#__ @ : decorator
#__ http://localhost/ => root url address 
#__ '/' 이게 루트
@app.route('/')
#__ http://localhost/hello => path가 /hello인 url address
@app.route('/hello')
@app.route('/world')
@app.route('/world/korea')
#__ 코드를 고치면 현재 돌아가고있는 서버가 restart된다.

def HelloWorld():
    return "Hello World"
#__ function으로 간편하게 구현 가능, standard는 class 객체 만들고 엄청 복잡하게 만들어져있다.

if __name__ == '__main__':
    app.debug = True 
#__ 디버깅을 하게 해주는 옵션 : true니깐 development 서버인 상태고 (개발할 때 사용하는 모드이고) 
#__ 배포를 할때는 false로 배포해라
    app.run(host='127.0.0.1', port=5000)
