# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# helper module
from helper_function import preprocessing
from helper_function import aug_bt
from helper_function import aug_rd
from helper_function import aug_ri
from helper_function import aug_sr
from helper_function import metrics
from helper_function import seed
import dataset

# model
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# tqdm
from tqdm import tqdm
tqdm.pandas() # progress

# other libraries
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import f1_score
import re
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


# start train.py
print('\n== start tuning.py ==')

# load data
X, y = dataset.make_dataset()


def objective(trial,X, y, model='LGBM',BT_e=False, BT_j=False, RD=False, RI=False, SR=False):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    k = kfold.get_n_splits(X, y)
    # print(f'split k : {k}')
    cnt_kfold = 1
    scores = []

    # == k-fold idx ==
    for tr_idx, val_idx in kfold.split(X, y):

        # seed everything
        seed.seed_everything(0)

        # == split ==
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, X_val, y_tr, y_val = pd.DataFrame(X_tr), pd.DataFrame(X_val), pd.DataFrame(y_tr), pd.DataFrame(y_val)

        if BT_e == True:

            # == tr aug ==
            out_en_y = pd.read_csv(f'./result/BTy{cnt_kfold}(en).csv')
            out_en_txt = pd.read_csv(f'./result/BTtxt{cnt_kfold}(en).csv')
            out_en_y = out_en_y[['label']]
            out_en_txt = out_en_txt[['txt']]
            # print('Done. (aug)')

            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            # print('Done. (concat)')

        if BT_e == False:

            X_tr_aug = X_tr.copy()
            y_tr_fin = y_tr.copy()

        if BT_j == True:

            # == tr aug ==
            out_en_y = pd.read_csv(f'./result/BTy{cnt_kfold}(jp).csv')
            out_en_txt = pd.read_csv(f'./result/BTtxt{cnt_kfold}(jp).csv')
            out_en_y = out_en_y[['label']]
            out_en_txt = out_en_txt[['txt']]

            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            # print('Done. (concat)')

        if BT_j == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()

        if RD == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_rd.RD(x))    # txt column 에 bt en 적용

            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/RDy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/RDtxt{cnt_kfold}.csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')

            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)

        if RD == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()

        if RI == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_ri.RI(x))    # txt column 에 bt en 적용

            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/RIy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/RItxt{cnt_kfold}.csv')  # save aug txt files ''' change '''

            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)

        if RI == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()

        if SR == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_sr.SR(x))    # txt column 에 bt en 적용

            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/SRy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/SRtxt{cnt_kfold}.csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')

            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)

        if SR == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()

        # == tr preprocessing ==
        X_tr_aug['txt'] = preprocessing.drop_duplicates(X_tr_aug['txt'])                    # 데이터 중복 제거
        X_tr_aug['txt'] = preprocessing.drop_null(X_tr_aug['txt'])                          # 결측치 제거 (=> 확인)
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.text_cleansing(x)) # 텍스트 킄렌징
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.text_tokenize(x))  # 토큰화
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.del_stopwords(x))  # 불용어 제거
        X_tr_fin = preprocessing.encoder_tf(X_tr_aug['txt'])                                # X_tr_fin & fit_transform tf-idf encoder 생성
        # print('Done. (tr preprocessing)')

        # == val preprocessing ==
        X_val['txt'] = preprocessing.drop_duplicates(X_val['txt'])
        X_val['txt'] = preprocessing.drop_null(X_val['txt'])
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.text_cleansing(x))
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.text_tokenize(x))
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.del_stopwords(x))
        X_val_fin = preprocessing.encoding_tf(X_val['txt'])
        # print('Done. (val preprocessing) \n')

        # == train model ==
        if model == 'LGBM':         # select model
            param_bo = {
                'n_estimators': trial.suggest_int("n_estimators", 40, 500),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, step=0.02),
                'subsample': trial.suggest_float('subsample', 0.6, 1, step=0.05),
                'num_leaves': trial.suggest_int("num_leaves", 20, 60),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1, step=0.1)
                }
            clf = LGBMClassifier(**param_bo, random_state=0)
        elif model == 'XGB':
            param_bo = {
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'n_estimators': trial.suggest_int("n_estimators", 40, 500),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, step=0.02),
                'subsample': trial.suggest_float('subsample', 0.6, 1, step=0.05),
                'max_leaves': trial.suggest_int("max_leaves", 200, 300),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.1)
                }
            clf = XGBClassifier(**param_bo, random_state=0)

        clf.fit(X_tr_fin, y_tr_fin) # , callbacks=[tqdm_callback])
        cnt_kfold += 1
        # == eval model ==
        val_pred = clf.predict(X_val_fin)
        scores.append(f1_score(y_val, val_pred))
    return np.mean(scores)


''' sample '''
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial,X, y, model='LGBM',BT_e=False, BT_j=False, RD=False, RI=False, SR=False), n_trials=72)

print(study.best_params)