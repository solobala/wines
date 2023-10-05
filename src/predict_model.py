# from keras.models import load_model

from keras.models import load_model
import pandas as pd
import click
from utils import read_pickle, write_pickle


@click.command()
@click.option('--type_', help="binary or cnn")
def main(type_: str):
    # предсказывать тип вина (красное или белое) можно с помощью
    # модели бинарной классификации или с помощью одного из выходов
    # нейросетевой модели. В первом случае точность выше,
    # но во втором - не требуется отдельная модель.
    # Здесь для предсказания типа вина используется 1 вариант
    if type_ == 'binary':
        model = read_pickle('models/binary')
        # Датасет для предикта по выбранной модели уже подготовлен
        X = pd.read_csv('data/processed/data_predict.csv', index_col=False)
        X = X.drop(columns=['type'])
        predictions = model.predict(X)
        write_pickle('reports/type_predictions.pkl', predictions)
    elif type_ == 'cnn':
        # Загружаем лучшую из двух моделей
        try:
            model = load_model('models/transform_model', compile=True)
            # Датасет для предикта уже подготовлен
            # (данные трансформированы с помощью build_features_cnn)
            X = pd.read_csv('data/processed/data_reg_transform_predict.csv',
                            index_col=False)
            X = X.drop(columns=['type', 'quality'])
            try:
                predictions = model.predict(X)
                write_pickle('reports/quality_predictions.pkl', predictions)
            except Exception as ex:
                print(ex)
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    main()
