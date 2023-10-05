import click
import pandas as pd
from pathlib import Path
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
import json
from utils import write_pickle
from settings import Settings
from app_settings import AppSettings
settings = Settings()
app_settings = AppSettings()


def train_model(df: pd.DataFrame,
                model: RandomForestClassifier
                ) -> tuple[pd.DataFrame, pd.Series,
                           pd.DataFrame, pd.Series,
                           RandomForestClassifier]:
    """
    Служебная функция. выполняет:
        - разделение на Train и test
        - обучение модели Бинарной классификации по типу вина 
        df- датафрейм, random_seed - фиксатор генератора случ.чисел, 
        model - модель классификатора,
        - сохранение модели
    """
    X = df.drop(columns=['type'])
    y = df['type']
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2,
                         random_state=settings.RANDOM_SEED)

    rf = model
    rf.fit(X_train, y_train)
    write_pickle(app_settings.TYPE_MODEL_FILE, rf)
    print('binary model fitted')
    return X, y, X_test, y_test, rf


def get_labels(data: pd.DataFrame):
    """ Get type & quality labels from whole dataset
    Args: pd.DataFrame
    Return: a tuple of 2 np.ndarrays"""
    type_wine = data.pop('type')
    type_wine = np.array(type_wine)
    quality = data.pop('quality')
    quality = np.array(quality)
    return (quality, type_wine)


def make_model(flag: str):
    """ Build 2 - outputs cnn model
    Args: flag: str, may be binary or cnn
    Return: Model"""
    if flag == 'cnn':
        # Вход - по числу признаков
        inputs = tf.keras.layers.Input(shape=(11,))
        # Полносвязные слои для создания нелинейности
        x = Dense(units=32, activation='relu')(inputs)
        x = Dense(units=32, activation='relu')(x)

        # Выход для определения типа вина (бинарная классификация)
        y_t_layer = Dense(units=1, activation='sigmoid', name='y_t_layer')(x)

        # Еще один полносвязный уровень для добавления выхода по качеству (x)
        quality_layer =\
            Dense(units=64, name='quality_layer', activation='relu')(x)

        # Выход для опеделения качества (регрессия)
        y_q_layer = Dense(units=1, name='y_q_layer')(quality_layer)

        model = Model(inputs=inputs, outputs=[y_q_layer, y_t_layer])

        optimizer = tf.keras.optimizers.Adam()

        # Для каждого выхода - свои метрики и функции потерь

        model.compile(optimizer=optimizer,
                      loss={'y_t_layer': 'binary_crossentropy',
                            'y_q_layer': 'mean_squared_error'
                            },
                      metrics={'y_t_layer': 'accuracy',
                               'y_q_layer': [
                                tf.keras.metrics.RootMeanSquaredError(),
                                tf.keras.metrics.MeanSquaredError(),
                                tf.keras.metrics.MeanAbsoluteError()]})
        print('cnn model builded')
    elif flag == 'binary':
        model = RandomForestClassifier(criterion='entropy',
                                       random_state=settings.RANDOM_SEED)
        print('binary model builded')
    return model


def scale_data(df, train_stats):
    # Нормализуем данные, используя mean и std от train. 
    return (df - train_stats['mean']) / train_stats['std']


def train_model_cnn(df: pd.DataFrame, flag: str) -> Model:
    """Fit & eval cnn model"""
    # Split data to train и test (80:20)
    train, test = train_test_split(df, test_size=0.2,
                                   random_state=settings.RANDOM_SEED)
    test_y_ = (test['quality'])

    # Split train to train и val (80:20).
    train, val = train_test_split(train, test_size=0.2,
                                  random_state=settings.RANDOM_SEED)

    # Scaling the 3 datasets.
    train_stats = train.describe()
    train_stats = train_stats.transpose()
    train_X = scale_data(train, train_stats).drop(columns=['type', 'quality'])
    test_X = scale_data(test, train_stats).drop(columns=['type', 'quality'])
    val_X = scale_data(val, train_stats).drop(columns=['type', 'quality'])
    # Get type и quality labels for train, test and validate.
    train_y = get_labels(train)
    test_y = get_labels(test)
    val_y = get_labels(val)

    model = make_model('cnn')
    model.fit(train_X, train_y,
              epochs=50, validation_data=(val_X, val_y))

    if flag == 'cnn':
        model.save(app_settings.CNN_MODEL_FILE)
    elif flag == 'transform':
        model.save(app_settings.TRANSFORM_MODEL_FILE)
    eval_model_cnn(model, test_X, test_y, test_y_, flag)
    return model


def eval_model(X, y, X_test, y_test, model, flag):
    """
    Служебная функция. выполняет:
        - расчет метрик качества модели бинарной классификации вина по типу в виде classification_report
        - построение confusion matrix
        - кросс-валидацию модели
        args:
        X: pd.DataFrame - датафрейм признаков,
        y: np.ndarray - массив целевой переменной,
        X_test: pd.DataFrame - тестовый датафрейм
        y_test: np.ndarray - тестовый массив целевой переменной,
        random_seed - фиксатор генератора случ.чисел, 
        model - модель классификатора,
        flag - признак печати результатов (True/False)
    """
    num_folds = 9
    scoring = 'r2'

    kfold = StratifiedKFold(n_splits=num_folds,
                            random_state=settings.RANDOM_SEED,
                            shuffle=True)

    y_pred = model.predict(X_test)
    target_names = ['red', 'white']
    # Отчет полностью
    report = classification_report(y_test, y_pred,
                                   target_names=target_names, output_dict=True)
    scoring = ['precision_macro', 'recall_macro']
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
    print(cv_results)
    write_pickle(Path('reports', 'cv_results'), cv_results)

    if flag:
        print("Classification report\n")
        print(report)
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 3))

    cm = confusion_matrix(y_test, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    cmp.plot(ax=ax, xticks_rotation='vertical')
    fig.savefig(str(Path('reports', "Confusion matrix.png")))

    return cv_results


def eval_model_cnn(model: Model,
                   X_test: pd.DataFrame,
                   y_test,
                   y_test_: pd.Series,
                   flag: str) -> None:
    """ Служебная функция. Выполняет оценку cnn модели
    как для классификакции по типу вина, так и для регрессии
    балла качества. Вычисляет:
    loss, wine_quality_loss, wine_quality_rmse,
    wine_quality_mse, wine_quality_mae, wine_type_loss, wine_type_accuracy.
    Args: model: Model,
          val_X: pd.DataFrame,
          val_y: pd. Series
    Return:  None
    """
    metrics = dict()
    (metrics['loss'],
     metrics['wine_quality_loss'],
     metrics['wine_type_loss'],
     metrics['wine_quality_rmse'],
     metrics['wine_quality_mse'],
     metrics['wine_quality_mae'],
     metrics['wine_type_accuracy']
     ) = model.evaluate(X_test, y_test)
    y_pred_q, _ = model.predict(X_test)
    y_test_ = np.array(y_test_)
    metrics['r2_score'] = r2_score(y_test_,  y_pred_q)

    for key, value in metrics.items():
        print(f'{key}: {value}')
    metrics_json = json.dumps(metrics)

    if flag == 'cnn':

        with open(app_settings.CNN_METRICS, 'w') as f:
            json.dump(metrics_json, f)

    elif flag == 'transform':

        with open(app_settings.TRANSFORM_METRICS, 'w') as f:
            json.dump(metrics_json, f)


@click.command()
@click.option('--type_', help="binary or cnn")
@click.option('--stage_', help="""To run functions from this module.
--make_model - to build & compile cnn model,
--train_model - fit binary classification model,
--train_model_cnn - fit cnn model,
--eval_model_cnn - to evaluate multy-input cnn model,
--eval_model - to evaluate binary classification model""")
def main(stage_: str, type_: str) -> None:
    if stage_ == 'make_model':
        _ = make_model(type_)

    elif stage_ == 'train_model':
        df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         index_col=False)
        X, y, X_test, y_test, rf = train_model(df,
                                               make_model(type_))
    elif stage_ == 'eval_model':
        df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         index_col=False)
        rf = make_model(type_)
        X, y, X_test, y_test, rf = train_model(df, rf)
        _ = eval_model(X, y, X_test, y_test, rf,
                       flag=True)

    elif stage_ == 'train_model_cnn':
        df = pd.read_csv(app_settings.PROCESSED_QUALITY_DATA_FILE,
                         index_col=False)
        _ = train_model_cnn(df, 'cnn')

    elif stage_ == 'train_model_transform':
        df = pd.read_csv(app_settings.PROCESSED_QUALITY_TRANSFORMED_DATA_FILE,
                         index_col=False)
        _ = train_model_cnn(df, 'transform')


if __name__ == '__main__':

    main()
