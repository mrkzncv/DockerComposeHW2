from flask import Flask
from flask_restx import Api
from flask_sqlalchemy import SQLAlchemy
from .ml_models import MLModelsDAO
import os
# from log import log

user = os.environ['POSTGRES_USER']  # postgres
password = os.environ['POSTGRES_PASSWORD']  # password
database = os.environ['POSTGRES_DB']  # postgres_db
host = os.environ['HOST']
port = os.environ['PORT']
DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'

app = Flask(__name__)
app.config['DATABASE_CONNECTION_URI'] = DATABASE_CONNECTION_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
api = Api(app)
db = SQLAlchemy(app)

models_dao = MLModelsDAO()

from app import views
