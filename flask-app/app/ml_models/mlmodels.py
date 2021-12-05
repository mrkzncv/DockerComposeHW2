from celery import Celery
import os

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

class MLModelsDAO:
    def __init__(self, ):
        self.ml_models = []
        self.counter = 0

    def get(self, id):
        """
        Функция обучает модель с заданным id и выдает предсказания
        :param id: integer: уникальный идентификатор модели
        :return: list: предсказания модели
        """
        f_name = None
        for model in self.ml_models:
            if model['id'] == id:
                task = celery.send_task('prediction', args=[model])
                return f'Task_id = {task.id}', 200
        if f_name is None:
            raise NotImplementedError('ml_model {} does not exist'.format(id))

    def create(self, data, is_new=True):  # пришли данные, надо присвоить id (для POST)
        """
        Обучение (переобучение) модели. На вход подается запрос на обучение модели и данные.
        Если у нас предусмотрена запрашиваемая функциональность, модель обучается и записывается в pickle
        с id модели в названии файла. Путь до файла записывается в json с ключом 'model_path'.
        :param data: json {'problem': 'classification', 'name': 'Random Forest', 'h_tune': False, 'X':x, 'y':y}
        :param is_new: boolean: новая ли модель или надо переобучать существующую
        :return: список обученных моделей
        """
        ml_model = data
        if (ml_model['problem'] in ['classification', 'regression']) and \
                (ml_model['name'] in ['Random Forest', 'SVM']):
            if is_new:
                self.counter += 1
                ml_model['id'] = self.counter
                self.ml_models.append(ml_model)
            task = celery.send_task('fit_model', args=[ml_model]) # ml_model - словарь со всеми параметрами
        else:
            raise NotImplementedError("""Сейчас доступны для обучения только classification and regression:
                                        Random Forest или SVM""")
        return f'Task_id = {task.id}', 200

    def update(self, id, data):
        """
        Функция либо переобучает модель, либо выдает ошибку, что такой модели ещё нет, надо создать новую
        :param id: integer: уникальный идентификатор модели
        :param data: json с новыми параметрами для модели
        :return: ничего не выдает
        """
        ml_model = None
        for model in self.ml_models:
            if model['id'] == id:
                ml_model = model  # json со старыми параметрами
        if ml_model is None:
            raise NotImplementedError('Такой модели ещё нет, надо создать новую')
        else:
            if (data['name'] == ml_model['name']) and (data['problem'] == ml_model['problem']):
                ml_model.update(data)  # кладу в них новые данные, 'X', 'y', 'h_tune'
                self.create(ml_model, is_new=False)  # переобучаю модель
            else:
                raise NotImplementedError('Такой модели ещё нет, надо создать новую')

    def delete(self, id):
        """
        Удаление модели по id
        :param id: integer: уникальный идентификатор модели
        :return: удаление модели из списка моделей
        """
        for model in self.ml_models:
            if model['id'] == id:
                self.ml_models.remove(model)
