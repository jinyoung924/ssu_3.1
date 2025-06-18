#!/usr/bin/env python3
#
# The *hello server* is an HTTP server that responds to a GET request by
# sending back a friendly greeting.  Run this program in your terminal and
# access the server at http://localhost:8000 in your browser.

from http.server import HTTPServer, BaseHTTPRequestHandler


class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # First, send a 200 OK response.
        self.send_response(200)

        # Then send headers.
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()

        # Now, write the response body.
        # self.wfile.write 메소드는 bytes data만 전송
        # encode() : string data를 utf-8 방식의 bytes로 변환하는 것을 말함
        #            (한글, 일본어, 중국어 등의 문자를 utf-8 방식의 bytes 체제로 변환)
        # encoding 방식 : utf-8(1~3 bytes), utf-16, ISO-8859 등,
        #                 python은 utf-8이 default encoding 방식임
        # encode의 목적은 web은 국제 표준이므로 browser와 web server가 영문이외에
        #                모든 language를 지원하기 위해서임
        # encode 예 : len('안녕') => 2 출력, len('안녕'.encode()) => 6 출력
        # cf : decode() : encode의 반대로, bytes data를 문자열로 변환
        self.wfile.write("Hello, HTTP!\n".encode())

if __name__ == '__main__':
    server_address = ('', 8000)  # Serve on all addresses, port 8000.
    httpd = HTTPServer(server_address, HelloHandler)
    httpd.serve_forever()
