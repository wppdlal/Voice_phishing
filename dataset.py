import pandas as pd

def make_dataset():

    # 데이터 로드
    df = pd.read_csv('./data/fin.csv')

    # X, y
    X = df['txt']
    y = df['label']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # print(f'X, y shape : {X.shape}, {y.shape}\n')

    return X, y

''' sample '''
# make_dataset()