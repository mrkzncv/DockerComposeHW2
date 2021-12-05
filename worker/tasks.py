import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from celery import Celery
import os
import json

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

# user = os.environ['POSTGRES_USER']  # POSTGRES
# password = os.environ['POSTGRES_PASSWORD']  # PASSWORD
# database = os.environ['POSTGRES_DB']  # POSTGRES_DB
# host = os.environ['HOST']
# port = os.environ['PORT']
# DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'

@celery.task(name='prediction')
def prediction(model):
    """
    Предсказание модели.
    """
    f_model_name = f"models/{model['problem']}_{model['name']}_{model['id']}.pickle"
    trained_model = pickle.load(open(f_model_name, 'rb'))
    predict_prob = trained_model.predict(np.array(model['X'])) # раньше делать array?
    return predict_prob.tolist()

@celery.task(name='get_metrics')
def get_metrics(model):
    """
    Выгрузка json со значениями параметров и метрик модели.
    """
    f_json_name = f"models/{model['problem']}_{model['name']}_{model['id']}.json"
    with open(f_json_name, "r") as fn:
        model_info = json.loads(fn.read())
    # model_info = json.load(open(f_json_name, 'rb'))
    return model_info['metric']

@celery.task(name='fit_model')
def fit_model(data):
    """
    Обучение (переобучение) модели. На вход подается запрос на обучение модели и данные.
    Если у нас предусмотрена запрашиваемая функциональность, модель обучается и записывается в pickle
    с id модели в названии файла. Путь до файла записывается в json с ключом 'model_path'.
    :param data: json {'problem': 'classification', 'name': 'Random Forest', 'h_tune': False, 'X':x, 'y':y}
    :return: список обученных моделей
    """
    ml_model = data
    x, y = np.array(ml_model['X']), np.array(ml_model['y']) # распаковать в таске
    if ml_model['problem'] == 'classification':
        best_model = classification(ml_model['name'], x, y, h_tune=ml_model['h_tune']) # обучение
        accuracy = accuracy_score(y, best_model.predict(x)) # надо записать в БД
        ml_model['metric'] = {'accuracy': accuracy}
        f_model_name = f"models/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.pickle"
        f_json_name = f"models/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.json"
        pickle.dump(best_model, open(f_model_name, 'wb'))
        with open(f_json_name, 'w') as fp:
            json.dump(ml_model, fp)
        # json.dump(ml_model, open(f_json_name, 'wb'))
    elif ml_model['problem'] == 'regression':
        best_model = regression(ml_model['name'], x, y, h_tune=ml_model['h_tune']) # обучение
        rmse = mean_squared_error(y, best_model.predict(x), squared=True ) # надо записать в БД
        ml_model['metric'] = {'rmse': rmse}
        f_model_name = f"models/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.pickle"
        f_json_name = f"models/{ml_model['problem']}_{ml_model['name']}_{ml_model['id']}.json"
        pickle.dump(best_model, open(f_model_name, 'wb'))
        with open(f_json_name, 'w') as fp:
            json.dump(ml_model, fp)
    return ml_model

def classification(model, x, y, h_tune=False):
    """
    :param model: название класса для модели классификации (строка) - "Random Forest" или "SVM".
    :param x: np.array(): выборка с признаками для обучения.
    :param y: np.array(): таргеты.
    :param h_tune: boolean: нужен ли подбор гиперпараметров или нет.
    :return: model(): обученная модель.
    """
    if model == 'Random Forest':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 4], 'max_features': ['auto', 'sqrt']}
        clf = RandomForestClassifier(random_state=0)
    elif model == 'SVM':
        param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        clf = SVC(random_state=0)

    if h_tune:
        clf_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        clf_cv.fit(x, y)
        return clf_cv.best_estimator_
    else:
        clf.fit(x, y)
        return clf

def regression(model, x, y, h_tune=False):
    """
    :param model: название класса для модели регрессии (строка) - "Random Forest" или "SVM".
    :param x: np.array(): выборка с признаками для обучения.
    :param y: np.array(): таргеты.
    :param h_tune: boolean: нужен ли подбор гиперпараметров или нет.
    :return: model(): обученная модель.
    """
    if model == 'Random Forest':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 4], 'max_features': ['auto', 'sqrt']}
        lr = RandomForestRegressor(random_state=0)
    elif model == 'SVM':
        param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        lr = SVR()

    if h_tune:
        lr_cv = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        lr_cv.fit(x, y)
        return lr_cv.best_estimator_

    else:
        lr.fit(x, y)
        return lr
