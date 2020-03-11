import pandas as pd
from sklearn import preprocessing

SCALE_FUNCTION = dict(
    [(attr, getattr(preprocessing, attr)) for attr in dir(preprocessing) if callable(getattr(preprocessing, attr))])


def load_data(path):
    '''
    Load data from csv to pandas DataFrame
    :param path (str): read file path
    :return df: pandas DataFrame
    '''

    df = pd.read_csv(path, sep='\t')
    return df


def great_primary_table(df):
    '''
    string column in the data frame df to int dataframe df_copy with sep=','
    :param df: pandas DataFrame

    :return:
    '''

    df_copy = df.copy()
    df_copy = pd.concat([df_copy.iloc[:, 0], df_copy.iloc[:, 1].str.split(',', expand=True)], axis=1)
    df_copy = df_copy.astype('int64')
    return df_copy


def abs_mean_diff(df):
    '''
    The function returns two series: indexes of maximum elements in rows and absolute values
    of deviations of the maximum element in a row from the average value in a column
    :param df: pandas DataFrame
    :return: idx (pandas series), result (pandas Series)
    '''

    idx = df.idxmax(axis=1)
    mean = df.mean()
    result = df.max(axis=1) - mean.loc[idx].reset_index(drop=True)
    return result.abs(), idx


def great_features_table(df, job_type, scale_func='StandardScaler', **settings):

    """
    The function creates a df_new dataframe whose columns are the standardized columns of the df dataframe.
    Standardization is carried out using one of the classes of the sklearn.preprocessing module.
    Also, the result of executing the abs_mean_diff function is included in the df_new dataframe

    :param df: pandas DataFrame
    :param job_type (int): Job type
    :param scale_func (str): sklearn.preprocessing  module class name (for example: StandardScaler)
    :param **settings: keywoard only arguments. Сlass parameters specified in scale_func
    :return df_new: result pandas DataFrame
    """

    scaler = SCALE_FUNCTION[scale_func](**settings)
    data_scale = scaler.fit_transform(df)
    df_new = pd.DataFrame(data_scale,
                          columns=['feature_' + str(job_type) + '_stand_' + str({i}) for i in range(df.shape[1])])
    abs_mean, idx = abs_mean_diff(df)
    abs_mean.rename('max_feature_' + str(job_type) + '_abs_mean_diff', inplace=True)
    idx.rename('max_feature_' + str(job_type) + '_index', inplace=True)
    df_new = pd.concat([df_new, idx, abs_mean], axis=1)
    return df_new

