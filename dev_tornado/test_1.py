import asyncio
import tornado
from tornado.web import Application as torn_app

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world__2")

def make_app():
    return torn_app([(r"/", MainHandler),])

async def main():
    app = make_app()
    app.listen(8877)
    await asyncio.Event().wait()

if __name__ == "__main__":
    print("[INFO]--Started Tonrnado App--->")
    asyncio.run(main())

