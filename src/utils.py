# Пользовательские функции
# Import libraries
import numpy as np  # linear algebra
from numpy.random import seed
import pickle
from pathlib import Path
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# from zipfile import ZipFile  # unzippung dataset

# Modelling algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# from sklearn import model_selection, metrics
# from sklearn.model_selection import  RandomizedSearchCV

# Modelling helpers
# from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# import scipy as sp
# from scipy import stats as st # При работе со статистикой
from scipy.stats import shapiro
from scipy.stats import f_oneway

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
# from IPython.core.display_trap import DisplayTrap

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
# %matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8, 6


# User functions for visualisation
def plot_histograms(df: pd.DataFrame, variables: list,
                    n_rows: int, n_cols: int) -> None:
    """Построение гистограмм признаков
    """
    fig = plt.figure(16, 12)
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title("Skew:" + str(round(float(df[var_name].skew()),))) # Distribution of df[var_name]
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
    fig.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, var: list,
                      target: list, **kwargs) -> None:
    """Построение распределений признаков"""
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()


def plot_categories(df: pd.DataFrame, cat: list,
                    target: list, **kwargs) -> None:
    """Построение барплота для категориальных признаков"""
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()


def plot_correlation_map(df: pd.DataFrame) -> None:
    """построение тепловой карты"""
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9},
                    ax=ax, annot=True, annot_kws={'fontsize': 12})


def draw_plot(X, names, type):
    """
    служебная функция для визуализации распределения признаков,
    построения боксплотов и графиков квантиль - квантиль.
    помогает визуально оценить нормальность распределения и
    наличие выбросов
    """
    n = len(names)
    seed(1)
    f, ax = plt.subplots(nrows=1, ncols=n, figsize=(22,2))
    i = 0
    for key, value in names.items():
        if type == 'distplot':
            distplot = sns.distplot(x=X[key],  ax=ax[i])
            distplot.set_xlabel('', fontsize=8)
            distplot.set_ylabel('', fontsize=8)
        elif type == 'boxplot':
            boxplot = sns.boxplot(x=X[key],  ax=ax[i])
            boxplot.set_xlabel('', fontsize=8)
            boxplot.set_ylabel('', fontsize=8)
        elif type == 'qqplot':
            qqplot(data=X[key], line='s', ax=ax[i])

        elif type == 'lineplot':
            lineplot = sns.lineplot(x=X[key], palette="tab10", linewidth=2.5, ax=ax[i])
            lineplot.set_xlabel('', fontsize=8)
            lineplot.set_ylabel('', fontsize=8)
        else:
            pass
        ax[i].set_title(value, fontsize=8)
        ax[i].minorticks_on()

        #  Определяем внешний вид линий основной сетки:
        ax[i].grid(which='major',
                   color='r',
                   linestyle='-')

        #  Определяем внешний вид линий вспомогательной
        #  сетки:
        ax[i].grid(which='minor',
                   color='k',
                   linestyle=':')

        i += 1


def get_transform(X, num_features):
    """
    Служебная функция для визуализации распределения числовых признаков
    после различной трансформации
    """
    for feature in num_features:

        X_tmp = X[feature]

        distributions =\
            [(f"{feature}: Unscaled data", X_tmp),
             (f"{feature}: standard",
             StandardScaler().fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: min-max",
             MinMaxScaler().fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: max-abs",
              MaxAbsScaler().fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: robust",
              RobustScaler(
                 quantile_range=(25, 75)
                 ).fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: power tr. (Yeo-Johnson)",
              PowerTransformer(
                 method="yeo-johnson"
                 ).fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: quantile tr.uniform",
                QuantileTransformer(
                   output_distribution="uniform"
                   ).fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: quantile tr. gaussian",
                QuantileTransformer(
                   output_distribution="normal"
                   ).fit_transform(X_tmp.to_numpy().reshape(-1, 1))),

             (f"{feature}: L2 normalizing",
              Normalizer().fit_transform(X_tmp.to_numpy().reshape(-1, 1)))]

        dist_dict = {}

        for item in distributions:
            dist_dict[item[0]] = item[0]

        transform_results =\
            pd.DataFrame(list(zip(distributions[0][1],
                                  distributions[1][1],
                                  distributions[2][1],
                                  distributions[3][1],
                                  distributions[4][1],
                                  distributions[5][1],
                                  distributions[6][1],
                                  distributions[7][1],
                                  distributions[8][1])),
                         columns=[distributions[0][0],
                                  distributions[1][0],
                                  distributions[2][0],
                                  distributions[3][0],
                                  distributions[4][0],
                                  distributions[5][0],
                                  distributions[6][0],
                                  distributions[7][0],
                                  distributions[8][0]])
        # визуализация распределения признака  после  трансформации
        draw_plot(transform_results, dist_dict, 'distplot')


# User functions for inference statistcs

def type_inf_stat_test(df: pd.DataFrame, feature: str) -> None:
    F, p = f_oneway(df[df.type == 'red'][feature],
                    df[df.type == 'white'][feature])
    if p <= 0.05:
        msg = 'Reject'
    else:
        msg = 'Accept'
    print('{}: F Statistic: {:.2f} \tp-value: {:.3f} \tNull Hypothesis: {}'.format(feature, F, p, msg))


def quality_inf_stat_test(df: pd.DataFrame, feature: str) -> None:
    F, p = f_oneway(df[df.quality_label == 'low'][feature],
                    df[df.quality_label == 'medium'][feature],
                    df[df.quality_label == 'high'][feature])
    if p <= 0.05:
        msg = 'Reject'
    else:
        msg = 'Accept'
    print('{}: F Statistic: {:.2f} \tp-value: {:.3f} \tNull Hypothesis: {}'.format(feature,F, p, msg))


def shapiro_test(df: pd.DataFrame, names: list) -> None:
    """
    Служебная функция для вывода информации по тесту Шапиро-Уилкса
    """
    for name in names:
        stat, p = shapiro(df[name])
        print(name)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('Предположительно нормальное распределение (Недостаточно данных, чтобы отвергнуть H0)')
        else:
            print('Отвергаем H0 о нормальности распределения')
        print()


def plot_model_var_imp(model, X, y) -> None:
    """Сортировка признаков по важности"""
    imp = pd.DataFrame(model.feature_importances_,
                       columns=['Importance'],
                       index=X.columns)
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[:10].plot(kind='barh')
    print(model.score(X, y))


def sigma_3(tmp: pd.DataFrame,
            names: list, percent: float) -> tuple[float, set]:
    """
    Служебная функция для поиска выбросов признака по правилу 3 сигм
    при нормальном или близком к ноормальному распределению
    tmp - датасет
    names - названия признаков
    percent - допустимый процент выбросов
    Возвращает % выбросов и их индексы
    """
    outliner_results = []

    for name in names:
        start_len = tmp.shape[0]
        result = dict()
        indices = []
        std = tmp[name].std()
        mean = tmp[name].mean()
        left_board = mean - 3 * std
        right_board = mean + 3 * std
        indices =\
            tmp.loc[
                (tmp[name] <= left_board) | (tmp[name] >= right_board)
                ].index.to_list()
        share = len(indices)/start_len*100
        result['indices'] = indices
        outliner_results.append(result)

    indices = set()

    for item in outliner_results:
        # объединяем выбросы по признакам
        indices = indices.union(item['indices'])

    share = (len(indices))/start_len * 100

    # возвращает % выбросов, множество индексов выбросов
    return share, indices


def interquartile_range(tmp: pd. DataFrame,
                        names: list, percent: float) -> tuple[float, set]:
    """
    Служебная функция для поиска выбросов признака по межквартильному размаху IQR
    при нормальном или близком к ноормальному распределению
    tmp - датасет
    names - названия признаков
    percent - допустимый процент выбросов
    Возвращает % выбросов и их индексы
    """
    outliner_results = []
    start_len = len(tmp)

    for name in names:
        result = dict()
        q1 = tmp[name].quantile(0.25)
        q3 = tmp[name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        indices =\
            tmp.loc[
                (tmp[name] <= lower_bound) | (tmp[name] >= upper_bound)
                ].index.to_list()
        share = len(indices)/start_len*100
        # result['feature']=name
        result['indices'] = indices
        # result['share'] = share
        outliner_results.append(result)

    indices = set()
    for item in outliner_results:
        indices = indices.union(item['indices'])
    share = (len(indices))/start_len * 100

    return share, indices


def percentile_range(tmp: pd.DataFrame,
                     names: list, percent: float) -> tuple[float, set]:
    """
    Служебная функция для поиска выбросов признака за пределами диапазона
    процентилей (0.5, 99.5)
        tmp - датасет
    names - названия признаков
    percent - допустимый процент выбросов
    Возвращает % выбросов и их индексы
    """
    outliner_results = []
    start_len = len(tmp)

    for name in names:
        result = dict()
        lower_limit, upper_limit = np.percentile(a=tmp[name], q=[0.5, 99.5])

        indices =\
            tmp.loc[
                (tmp[name] <= lower_limit) | (tmp[name] >= upper_limit)
                ].index.to_list()
        share = len(indices)/start_len*100
        result['indices'] = indices
        outliner_results.append(result)

    indices = set()
    for item in outliner_results:
        indices = indices.union(item['indices'])
    share = (len(indices))/start_len * 100

    return share, indices


def mahalanobis(x=None, data=None, cov=None):
    """
    Функция для расчета расстояния Maхаланобиса
    Расстояние Махаланобиса — это расстояние между двумя точками в
    многомерном пространстве. Оно часто используется для поиска выбросов
    в статистическом анализе, включающем несколько переменных
    https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%9C%D0%B0%D1%85%D0%B0%D0%BB%D0%B0%D0%BD%D0%BE%D0%B1%D0%B8%D1%81%D0%B0
    """
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def get_learn(df: pd.DataFrame, random_seed: int,
              method: str, model: RandomForestClassifier, flag: bool):
    """
    Служебная функция. выполняет:
        - разделение на Train и test
        - обучение модели
        - расчет метрик качества classification_report
        возвращает название метода удаления выбросов,
        classification_report в виде словаря и отдельно - accuracy
        df- датафрейм, random_seed - фиксатор генератора случ.чисел,
        method - название метода поиска выбросов, model - модель классификатора,
        flag - признак печати результатов (True/False)
    """
    num_folds = 9
    scoring = 'r2'

    X = df.drop(columns=['type'])
    y = df['type']
    kfold = StratifiedKFold(n_splits=num_folds, random_state=random_seed,
                            shuffle=True)
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2,
                         random_state=random_seed)

    rf = model
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    target_names = ['red', 'white']
    if flag:
        report = classification_report(
            y_test, y_pred, target_names=target_names)
        print(f"Classification report {method}\n")
        print(
            f"{report}\n"
        )
        # Confusion matrix

        fig, ax = plt.subplots(figsize=(6, 3))

        cm = confusion_matrix(y_test, y_pred)
        cmp = ConfusionMatrixDisplay(cm, display_labels=target_names)
        # plt.rcParams.update({'font.size': 10})
        # plt.tick_params(axis='both', which='major', labelsize=10)
        cmp.plot(ax=ax, xticks_rotation='vertical')
        plt.show()
    # Отчет полностью
    report = classification_report(y_test, y_pred,
                                   target_names=target_names,
                                   output_dict=True)
    a = report['accuracy']
    scoring = ['precision_macro', 'recall_macro']
    cv_results = cross_validate(rf, X, y, cv=kfold, scoring=scoring)

    return method,  report, a, cv_results, rf


def write_pickle(target: str, to_save: object) -> None:
    """ Save data to .pkl

    Args:
        target: str - a path to file
        to_save: object - object to save
    Return:
        None

    Raise:
        Exception("Failed to write to <a pickle file name>)
    """
    try:

        with open(str(Path(Path.cwd(), target)), 'wb') as f:
            pickle.dump(to_save, f)

    except Exception as ex:
        raise ex


def read_pickle(target: str):
    """ To read data from .pkl file

    Args:
        target: str -  a path to file

    Return:
        object, readed from pickle file

    Raise:
        Exception("Failed to read from <a pickle file name>)
    """

    try:
        with open(target, 'rb') as f:
            to_load = pickle.load(f)
        return to_load

    except FileNotFoundError as ex:
        raise ex
