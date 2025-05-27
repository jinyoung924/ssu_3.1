#!/usr/bin/env python3
#
# This is the solution code for the *echo server*.

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class PostFormHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. How long was the message?
        # 1.1 POST 방식인 경우 request header에 'Content-length'가 추가되고,
        # 1.2 request body에 있는 data 전체 크기 정보를 갖고 있음
        # 1.3 headers 정보 가져올 때 get method 사용
        #     - get 메소드 대신에 self.headers['content-length'] 사용 가능
        #       (=> 하지만 'Content-length'라는 key가 없으면 Error발생)
        #     - get 메소드를 쓰는 이유는 'Content-length'라는 key가 없더라도 Error 발생않함
        length = int(self.headers.get('Content-length', 0))

        # 2. Read the correct amount of data from the request.
        # data : string data type
        data = self.rfile.read(length).decode()

        # 3. Extract the "message" field from the request data.
        # 3.1 parse_qs의 return data type은 dictionary임  {'id': ['value1'], 'password': ['value2']}
        # - query string의 동일 key에 해당하는 value가 multiple이 될 수있어, value는 list data type임
        #  [0]의 의미 (message key의 첫번째 value값)
        magic = parse_qs(data)["magic"][0]
        secret = parse_qs(data)["secret"][0]

        # Send the "message" field back as the response.
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()

        self.wfile.write(('magic' + ' = ' + magic + '\n').encode())
        self.wfile.write(('secret' + ' = ' + secret + '\n').encode())


if __name__ == '__main__':
    server_address = ('', 9999)  # Serve on all addresses, port 9999.
    httpd = HTTPServer(server_address, PostFormHandler)
    httpd.serve_forever()
