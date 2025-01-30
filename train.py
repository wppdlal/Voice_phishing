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
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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

# model list
model_list = []


def train(X, y, model='RF',BT_e=False, BT_j=False, RD=False, RI=False, SR=False):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    k = kfold.get_n_splits(X, y)
    # print(f'split k : {k}')
    cnt_kfold = 1
    best_acc, best_f1, best_recall = 0, 0, 0 # to save best model
    cumsum_acc, cumsum_f1, cumsum_recall = 0, 0, 0

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
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if BT_e == False:

            X_tr_aug = X_tr.copy()
            y_tr_fin = y_tr.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')

        if BT_j == True:

            # == tr aug ==
            out_en_y = pd.read_csv(f'./result/BTy{cnt_kfold}(jp).csv')  
            out_en_txt = pd.read_csv(f'./result/BTtxt{cnt_kfold}(jp).csv') 
            out_en_y = out_en_y[['label']]
            out_en_txt = out_en_txt[['txt']]
            # print('Done. (aug)')
            
            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if BT_j == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')    

        if RD == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_rd.RD(x))    # txt column 에 bt en 적용
            print(f'raw shape : {out.shape}')
            print(f'aug shape : {out_en.shape}')
            '''
            print(f'raw sample :')
            print(f'{out.iloc[0]}')
            print(f'aug sample :')
            print(f'{out_en.iloc[0]}')
            '''
            
            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/RDy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/RDtxt{cnt_kfold}.csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')
            
            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if RD == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')  

        if RI == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_ri.RI(x))    # txt column 에 bt en 적용
            print(f'raw shape : {out.shape}')
            print(f'aug shape : {out_en.shape}')
            '''
            print(f'raw sample :')
            print(f'{out.iloc[0]}')
            print(f'aug sample :')
            print(f'{out_en.iloc[0]}')
            '''
            
            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/RIy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/RItxt{cnt_kfold}.csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')
            
            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if RI == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')  

        if SR == True:

            # == tr aug ==
            out = pd.concat([X_tr, y_tr], axis=1).copy()  # train copy

            out_en = out[out['label'] == 1].copy()                                         # train 중 피싱 데이터만 copy
            out_en['txt'] = out_en['txt'].progress_apply(lambda x : aug_sr.SR(x))    # txt column 에 bt en 적용
            print(f'raw shape : {out.shape}')
            print(f'aug shape : {out_en.shape}')
            '''
            print(f'raw sample :')
            print(f'{out.iloc[0]}')
            print(f'aug sample :')
            print(f'{out_en.iloc[0]}')
            '''
            
            out_en_y = out_en['label']  # bt result y
            out_en_txt = out_en['txt']  # bt result txt
            out_en_y, out_en_txt = pd.DataFrame(out_en_y), pd.DataFrame(out_en_txt)
            out_en_y.to_csv(f'./result/SRy{cnt_kfold}.csv')      # save aug y files   ''' change '''
            out_en_txt.to_csv(f'./result/SRtxt{cnt_kfold}.csv')  # save aug txt files ''' change '''
            # print('Done. (aug)')
            
            # tr concat : origin + aug
            X_tr_aug = pd.concat([X_tr_aug, out_en_txt], axis=0, ignore_index=True)  # fin train txt (raw + aug)
            y_tr_fin = pd.concat([y_tr_fin, out_en_y], axis=0, ignore_index=True)    # fin train y   (raw + aug)
            print(f'concat shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')
            print(f'raw + aug shape : {X_tr_aug.shape}')
            # print('Done. (concat)')
        
        if SR == False:

            X_tr_aug = X_tr_aug.copy()
            y_tr_fin = y_tr_fin.copy()
            print(f'shape : ({X_tr_aug.shape}, {y_tr_fin.shape}) \n')   

        # == tr preprocessing ==
        X_tr_aug['txt'] = preprocessing.drop_duplicates(X_tr_aug['txt'])                    # 데이터 중복 제거
        print('Done. (drop duplicates)')
        X_tr_aug['txt'] = preprocessing.drop_null(X_tr_aug['txt'])                          # 결측치 제거 (=> 확인)
        print('Done. (drop null)')
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.text_cleansing(x)) # 텍스트 킄렌징
        print('Done. (text cleansing)')
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.text_tokenize(x))  # 토큰화
        print('Done. (tokenization)')
        X_tr_aug['txt'] = X_tr_aug['txt'].apply(lambda x : preprocessing.del_stopwords(x))  # 불용어 제거
        print('Done. (del stopwords)')
        print(f'raw + aug shape : {X_tr_aug.shape}')
        X_tr_fin = preprocessing.encoder_tf(X_tr_aug['txt'])                                # X_tr_fin & fit_transform tf-idf encoder 생성
        print('Done. (train preprocessing)')
        # print('Done. (tr preprocessing)')

        # == val preprocessing ==
        X_val['txt'] = preprocessing.drop_duplicates(X_val['txt'])
        print('Done. (drop duplicates)')
        X_val['txt'] = preprocessing.drop_null(X_val['txt'])
        print('Done. (drop null)')
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.text_cleansing(x))
        print('Done. (text cleansing)')
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.text_tokenize(x))
        print('Done. (tokenization)')
        X_val['txt'] = X_val['txt'].apply(lambda x : preprocessing.del_stopwords(x))
        print('Done. (del stopwords)')
        X_val_fin = preprocessing.encoding_tf(X_val['txt'])
        print('Done. (valid preprocessing)')
        # print('Done. (val preprocessing) \n')

        # == train model ==
        if model == 'LGBM':         # select model
            clf = LGBMClassifier()
        elif model == 'XGB':
            clf = XGBClassifier()
        elif model == 'RF':
            clf = RandomForestClassifier()

        clf.fit(X_tr_fin, y_tr_fin) # , callbacks=[tqdm_callback])

        # == eval model ==
        pred = clf.predict(X_val_fin)
        acc, f1, recall = metrics.metrics(y_val, pred)
        print(f'acc, f1, recall : {acc}, {f1}, {recall}')
        cumsum_acc += acc
        cumsum_f1 += f1
        cumsum_recall += recall
        time.sleep(0.5)
        # print('Done. (train/eval model) \n')

        # == check best model == 
        if f1 > best_f1:
            # == save best model and vectorizer ==
            best_f1 = f1
            best_acc = acc
            best_recall = recall
            best_model = clf
            pickle.dump(best_model, open('./result/best_f1_model (rf,sr).pkl', 'wb')) # save best model ''' change '''
            preprocessing.save_encoder_tf(X_tr_aug['txt']) # save best encoder
            
        cnt_kfold += 1

    pickle.dump(model_list, open('model_list.pkl', 'wb'))
    print(f'\n')
    print(f'best acc, f1, recall : {best_acc}, {best_f1}, {best_recall}')
    print(f'avg acc, f1, recall : {round(cumsum_acc / 5, 5)}, {round(cumsum_f1 / 5, 5)}, {round(cumsum_recall / 5, 5)}')
    print(f'check best model : ./result/best_f1_model (rf,sr).pkl') # 저장된 best model & encoder 이름 바꾸기
    return


''' sample '''
# train(X, y, 'LGBM', BT=True)
# train(X, y, 'LGBM', BT=False)
# train(X, y, 'XGB', BT=False)
#train(X, y, 'XGB', BT=True)
train(X, y, model='RF',BT_e=False, BT_j=False, RD=False, RI=False, SR=False)
