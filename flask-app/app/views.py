from flask_restx import Resource, fields
from app import api, models_dao, db
import logging
from celery import Celery
import os

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

log = logging.getLogger(__name__)

# шаблон с описанием сущности
# ml_models_desc = api.model('ML models', {'id': fields.Integer,
#                                          'problem': fields.String,
#                                          'name': fields.String,
#                                          # 'accuracy': fields.Float,
#                                          'h_tune': fields.Boolean,
#                                          'X': fields.List,  # ??
#                                          'y': fields.List,  # ??
#                                          'h_params': fields.List,
#                                          'prediction': fields.List})

implemented_models = {'classification': ['Random Forest', 'SVM'],
                      'regression': ['Random Forest', 'SVM']}


@api.route('/api/ml_models')
class MLModels(Resource):

    @staticmethod
    def get():
        """
        Возвращает доступные классы и информацию об обученных моделях в виде json:
            {'problem': 'classification', 'name': 'Random Forest', 'h_tune': False, 'X':x, 'y':y, 'id': 1}
        Путь до модели на воркере можно всегда получить по ключам 'problem', 'name' и 'id':
            f_name: models/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.pickle
        """
        return [implemented_models, models_dao.ml_models]

    @staticmethod
    # @api.expect(ml_models_desc) # нужно проверить то, что отдает клиент, на валидность
    def post():
        """
        Обучение новой модели.
        """
        return models_dao.create(api.payload)

    def put(self):  # update?
        pass

    def delete(self):
        pass

@api.route('/results/<task_id>') # предсказания , methods=['GET']
class Results(Resource):

    def get(self, task_id):
        """
        Получить предсказания модели.
        """
        res = celery.AsyncResult(task_id)
        if res.status == 'PENDING':
            return str(res.state), 200
        else:
            return res.get(), 200

@api.route('/metrics/<int:id>')
class MLModelsMetrics(Resource):

    def post(self, id):
        """
        Добавляем в базу данных информацию о метриках обученной модели по её id.
        """
        for model in models_dao.ml_models:
            if model['id'] == id:
                task = celery.send_task('get_metrics', args=[model])
                res = celery.AsyncResult(task.id).get()
                metric, metric_value = list(res.items())[0] # .items()
                model_id = id
                problem = model['problem']
                model_name = model['name']
                h_tune = model['h_tune']
                model_metrics = ModelsDB(model_id, problem, model_name, h_tune, metric, metric_value)
                db.session.add(model_metrics)
                db.session.commit()
                return f'Метрики по модели {id}, {problem}, {model_name} добавлены в Базу Данных'

    def get(self, id):
        """
        Возвращаем информацию о метриках обученной модели по её id
        """
        model_info = ModelsDB.query.filter_by(model_id=id).all()[0]
        return {'problem': model_info.problem, 'name': model_info.model_name,
                'metric':model_info.metric, 'metric_value': model_info.metric_value}, 200

    def put(self, id):
        """
        Обвновление данных о метриках после переобучения модели по её id
        """
        for model in models_dao.ml_models:
            if model['id'] == id:
                task = celery.send_task('get_metrics', args=[model])
                res = celery.AsyncResult(task.id).get()
                metric, new_metric_value = list(res.items())[0]
                model_id = id
                old_model_metrics = ModelsDB.query.filter_by(model_id=model_id,
                                    problem=model['problem'], model_name=model['name'],
                                    h_tune=model['h_tune'], metric=metric).all()[0]
                old_model_metrics.value = new_metric_value
                db.session.commit()
                return f'Метрики по модели {id}, {model["problem"]}, {model["name"]} обновлены в Базе Данных'

    def delete(self, id):
        """
        Удалить информацию по метрикам модели из Базы Данных
        """
        ModelsDB.query.filter_by(model_id=id).delete()
        db.session.commit()
        return f'Данные о метриках модели удалены из Базы Данных'

@api.route('/api/ml_models/<int:id>')
class MLModelsID(Resource):

    @staticmethod
    def get(id):
    # log.info(f'id = {id}\n type(id) = {type(id)}')
        try:
            return models_dao.get(id)
        except NotImplementedError as e:
            api.abort(404, e)

    @staticmethod
    # @api.expect(ml_models_desc)
    def put(id):  # update
        """
        Переобучение модели на новых данных по id.
        """
        return models_dao.update(id, api.payload)

    @staticmethod
    # @api.expect(ml_models_desc)
    def delete(id):
        """
        Удаление данных о модели.
        """
        models_dao.delete(id)
        return '', 204

class ModelsDB(db.Model):
    """
    Структура таблицы в базе данных с описанием типов полей.
    :param id: инкрементальный id
    :param model_id: id модели
    :param problem: classification или regression
    :param model_name: Random Forest или SVM
    :param h_tune: True или False - нужен ли подбор гиперпараметров
    :param metric: метрика качества. Accuracy для классификации, RMSE для регрессии
    :param metric_value: значение метрики качества
    """
    __tablename__ = 'mao_models'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer)
    problem = db.Column(db.String(15))
    model_name = db.Column(db.String(15))
    h_tune = db.Column(db.Boolean)
    metric = db.Column(db.String(5))  # rmse for regression, accuracy for classification
    metric_value = db.Column(db.Float)

    def __init__(self, model_id, problem, model_name,
                 h_tune, metric, metric_value):
        self.model_id = model_id
        self.problem = problem
        self.model_name = model_name
        self.h_tune = h_tune
        self.metric = metric
        self.metric_value = metric_value