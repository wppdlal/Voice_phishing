# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# helper module
from helper_function import preprocessing
from helper_function import aug_bt
from helper_function import metrics
from helper_function import seed
import dataset

# tqdm
from tqdm import tqdm
tqdm.pandas() # progress

# other libraries
import pandas as pd
import numpy as np
import re
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


# start train.py
print('\n== start train.py ==')

# load data
X, y = dataset.make_dataset()


def train(X, y, BT=True):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    k = kfold.get_n_splits(X, y)
    # print(f'split k : {k}')
    cnt_kfold = 1

    # == k-fold idx ==
    for tr_idx, val_idx in kfold.split(X, y):

        # seed everything
        seed.seed_everything(0)

        # k-fold
        print(f'\n== K-FOLD {cnt_kfold} ==\n')
        print(f'TRAIN : {tr_idx}')      # kfold train index
        print(f'VALID : {val_idx} \n')  # kfold test index

        # == split ==
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, X_val, y_tr, y_val = pd.DataFrame(X_tr), pd.DataFrame(X_val), pd.DataFrame(y_tr), pd.DataFrame(y_val)

        ''' del '''
        # X_tr = X_tr.iloc[:2]
        # y_tr = y_tr.iloc[:2]
        # print(X_tr.shape, y_tr.shape)
        ''' del '''

        if BT == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy
            
            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy (en)
            out_jp = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy (jp)

            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_bt.BT_ko2en(x))    # txt column 에 bt en 적용
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_bt.BT_en2ko(x))    # txt column 에 bt ko 적용

            out_jp['txt'] = out_jp['txt'].progress_apply(lambda x : aug_bt.BT_ko2jp(x))    # txt column 에 bt jp 적용
            out_jp['txt'] = out_jp['txt'].progress_apply(lambda x : aug_bt.BT_jp2ko(x))    # txt column 에 bt ko 적용

            ## en, jp concat!
            #out_en['txt'] =  pd.concat([out_en['txt'], out_jp['txt']], ignore_index=True)  # en, jp 결과 합칠때만 활성화
            
            #print(f'raw shape : {out.shape}')
            #print(f'aug shape : {out_en.shape}')
            '''
            print(f'raw sample :')
            print(f'{out.iloc[0]}')
            print(f'aug sample :')
            print(f'{out_en.iloc[0]}')
            '''
            
            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_jp_y = out_jp['label']  # bt result y
            out_jp_txt = out_jp['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_jp_y, out_jp_txt = pd.DataFrame(out_jp_y), pd.DataFrame(out_jp_txt)
            out_en_y.to_csv(f'./result/BTy{cnt_kfold}(en).csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/BTtxt{cnt_kfold}(en).csv')  # save aug txt files ''' change '''
            out_jp_y.to_csv(f'./result/BTy{cnt_kfold}(jp).csv')      # save aug y files   ''' change '''
            out_jp_txt.to_csv(f'./result/BTtxt{cnt_kfold}(jp).csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')
            
            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if BT == False:

            X_tr_aug = X_tr.copy()
            y_tr_fin = y_tr.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
          
    return


''' sample '''
train(X, y, BT=True)