#!/usr/bin/env python3
#
# This is the solution code for the *echo server*.

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class LoginHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # First, send a 200 OK response.
        self.send_response(200)

        # Then send headers.
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()

        # print(parts) 하면 아래와 같이 출력됨
        #    (url이 'https://www.google.com/search?q=gray+squirrel&tbm=isch' 인 경우)
        #     ParseResult(scheme='https', netloc='www.google.com', path='/search',
        #                   params='', query='q=gray+squirrel&tbm=isch', fragment='')
        parts = urlparse(self.path)
        try:
            # q = dict([p.split('=') for p in parts[4].split('&')])
            q = dict([p.split('=') for p in parts.query.split('&')])
        except:
            q = {}
        for k in q.keys():
            self.wfile.write((k + ' = ' + q[k] + '\n').encode())


if __name__ == '__main__':
    server_address = ('', 8000)  # Serve on all addresses, port 8000.
    httpd = HTTPServer(server_address, LoginHandler)
    httpd.serve_forever()
