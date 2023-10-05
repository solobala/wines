import click
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import json
from settings import Settings
from app_settings import AppSettings
import warnings
warnings.filterwarnings('ignore')
settings = Settings()
app_settings = AppSettings()


def make_login() -> None:
    """ Create json file with authentication data for Kaggle from .env"""
    user_name = settings.USER_NAME
    key = settings.KEY
    # создаем словарь с данными для файла kaggle.json
    kaggle_info = {
      "username": user_name,
      "key": key}

    # преобразуем словарь в строку JSON
    json_info = json.dumps(kaggle_info)

    # создаем файл kaggle.json и записываем в него строку JSON
    # with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    try:
        # with open(Path(".kaggle", "kaggle.json"), "w") as f:
        with open(Path(app_settings.KAGGLE_CONFIG_DIR,
                       "kaggle.json"), "w") as f:
            f.write(json_info)
        print("kaggle.json was created")
    except Exception:
        print("Failed!")


def make_dataset(file_path: str,
                 zip_name: str, file_name: str) -> pd.DataFrame:
    """Функция для скачивания и распаковки датасета с kaggle
    """
    # Загрузка файла аутентификации
    import kaggle
    kaggle.api.authenticate()

    # Установка пути для загрузки данных
    # data_path = Path('data', 'raw')
    data_path = app_settings.RAW_DATA_FOLDER

    # Скачивание данных по ссылке
    kaggle.api.dataset_download_files(file_path, path=data_path)
    with ZipFile(zip_name, 'r') as zip_file:
        zip_file.extractall(data_path)
    df = pd.read_csv(Path(data_path, file_name), index_col=False)
    print('Файл загружен!')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление пропусков, удаление дубликатов
    Args: df: pd.DataFrame
    Return: df: pd.DataFrame"""
    d = df[(pd.isnull(df['fixed acidity']) |
            pd.isnull(df['pH']) |
            pd.isnull(df['volatile acidity']) |
            pd.isnull(df['sulphates']) |
            pd.isnull(df['citric acid']) |
            pd.isnull(df['residual sugar']) |
            pd.isnull(df['chlorides']))]

    if len(d)/df.shape[0]*100 < 5:
        # data_interim_path = Path('data', 'interim')
        df = df[~(pd.isnull(df['fixed acidity']) |
                pd.isnull(df['pH']) |
                pd.isnull(df['volatile acidity']) |
                pd.isnull(df['sulphates']) |
                pd.isnull(df['citric acid']) |
                pd.isnull(df['residual sugar']) |
                pd.isnull(df['chlorides']))]

        df.drop_duplicates(inplace=True)
        df.to_csv(Path(app_settings.INTERIM_DATA_FOLDER, 'cleaned_wines.csv'),
                  index=False, columns=df.columns)
        print('Очистка завершена')
    return df


@click.command()
@click.option('--stage_', help="""To run functions from this module.
--make_login - to use authentication data for Kaggle,
--make_dataset - to download dataset to data/raw/        
--clean_data - to clean the raw dataset (drop duplicates & nan)
--all - run all stages""")
def main(stage_: str) -> None:
    if stage_ == 'make_login':
        make_login()
    elif stage_ == 'make_dataset':
        df = make_dataset('rajyellow46/wine-quality',
                          f'{app_settings.RAW_DATA_FOLDER}/wine-quality.zip',
                          'winequalityN.csv')
    elif stage_ == 'clean_data':
        df = make_dataset('rajyellow46/wine-quality',
                          f'{app_settings.RAW_DATA_FOLDER}/wine-quality.zip',
                          'winequalityN.csv')
        df = clean_data(df)
    elif stage_ == 'all':
        make_login()
        _ = clean_data(
            make_dataset('rajyellow46/wine-quality',
                         f'{app_settings.RAW_DATA_FOLDER}/wine-quality.zip',
                         'winequalityN.csv'))


if __name__ == '__main__':
    main()
