# from config import HOST, PORT
import sys
import os

def start():
    from app import app, db
    db.create_all()
    app.run(debug=True, host=os.environ['HOST'], port=os.environ['PORT'])

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.getcwd()))
    start()


