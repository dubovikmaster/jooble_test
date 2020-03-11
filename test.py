import mymodule
import pandas as pd

path_read = 'C:/Users/Tom/DS/Jooble/data/test.tsv'          # your read file path
path_save = 'C:/Users/Tom/DS/Jooble/data/test_proc.tsv'     # your write file path


def test(path_read, path_save):
    try:
        df = mymodule.load_data(path_read)
        df = mymodule.great_primary_table(df)
        job_type = df.iloc[0, 1]
        feature_table = mymodule.great_features_table(df.iloc[:, 2:], job_type)
        result = pd.concat([df.iloc[:, 0], feature_table], axis=1)
        result.to_csv(path_save, sep='\t')
    except Exception as e:
        print('ERROR:', str(e), sep=' ')

test(path_read, path_save)