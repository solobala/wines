import click
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import QuantileTransformer
from settings import Settings
from app_settings import AppSettings
settings = Settings()
app_settings = AppSettings()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Конструирование признаков
    Args: pd.DataFrame
    Return: pd.DataFrame"""
    # Замена категориальной переменной type (тип вина) на числовую (0 и 1)
    replace_dict = {'red': 0, 'white': 1}
    df['type'] = df['type'].map(replace_dict)

    # определяем колонки с числовыми признаками
    numeric_features =\
        df.drop(columns=['type', 'quality']).columns.tolist()

    # Подготовка датафрейма для бинарной классификации вин по типу
    X = df[numeric_features]

    y = df['type']
    X_reg = pd.concat([X, y], axis=1)

    # sm = SMOTE(random_state=42, k_neighbors=5)
    sm = SMOTE(random_state=42, k_neighbors=4)
    X, y = sm.fit_resample(X, y)
    y_tmp = y.to_frame()
    data = pd.concat([X, y_tmp], axis=1)  # датафрейм после оверсэмплинга.
    data.to_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                index=False, columns=data.columns)

    # Подготовка датафрейма для регрессии оценки качества
    y_reg = df['quality']
    X_reg, y_reg = sm.fit_resample(X_reg, y_reg)

    y_tmp_reg = y_reg.to_frame()
    data_reg = pd.concat([X_reg, y_tmp_reg], axis=1)
    data_reg.to_csv(app_settings.PROCESSED_QUALITY_DATA_FILE, index=False,
                    columns=data_reg.columns)
    print('feature engineering complete')
    return data, data_reg


def build_features_cnn(df) -> pd.DataFrame:
    # Получаем сбалансированный по SMOTE датасет для регрессии
    data, data_reg = build_features(df)
    # Выполняем трансформацию числовых признаков с помощью QuantileTransformer
    numeric_features = data_reg.drop(columns=['type', 'quality'])
    qt = QuantileTransformer(n_quantiles=10, random_state=42)
    names = []
    for name in numeric_features:
        names.append(
            qt.fit_transform(data_reg[name].to_numpy().reshape(-1, 1)))
    names = [arr.reshape(1, -1)[0] for arr in names]
    X_transform = pd.DataFrame(names).transpose()
    X_transform.columns = numeric_features.columns

    # Снова собираем все признаки для регрессии и обе целевые переменные
    # в датафрейм data_reg_transform
    data_transform = pd.concat([X_transform, data_reg['type']], axis=1)

    data_reg_transform = pd.concat([data_transform,
                                    data_reg['quality']], axis=1)

    data_reg_transform.to_csv(
        app_settings.PROCESSED_QUALITY_DATA_FILE_TO_PREDICTION,
        index=False, columns=data_reg_transform.columns)
    print('feature engineering complete')
    return data_reg_transform


@click.command()
@click.option('--stage_', help="""To run functions from this module.
--stage_=='build_features' - feature engineering for binary classification
& regression,
--stage_=='build_features_cnn' - feature engineering
              for cnn multy-outputs model""")
def main(stage_: str):
    df = pd.read_csv('data/interim/cleaned_wines.csv', index_col=False)
    if stage_ == 'build_features':
        _, _ = build_features(df)
    elif stage_ == 'build_features_cnn':
        _ = build_features_cnn(df)


if __name__ == '__main__':
    main()
