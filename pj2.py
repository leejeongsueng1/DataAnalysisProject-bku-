# 파이썬 기본모듈
import os
import warnings

# numpy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 전처리 모듈 import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# 학습 모델 import
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# voting관련 학습 모델 추가 import 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# 평가함수
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 데이터 로드
data = pd.read_csv('data/train.csv')


# csv 파일 로드함수
def load_data(path):
    return pd.read_csv(path)


# Null 및 여러 정보들을 보여주는 함수
def analisys(data,axis=0):
    if axis==0:
        #file open
        f = open('analysis.txt','w')

        f.write('------------------Data Analysis--------------\n\n')

        #dtypes
        f.write('-Types-----------------------\n')
        f.write(data.dtypes.to_string())
        f.write('\n\n')

        #Shapes
        f.write('-Shapes-----------------------\n')
        f.write(str(data.shape))
        f.write('\n\n')

        #Nulls
        f.write('-Nulls-----------------------\n')
        f.write(data.isna().sum().to_string())
        f.write('\n\n')
    else:
        #file open
        f = open('analysis_after_Encoding.txt','w')

        f.write('------------------Data Analysis--------------\n\n')

        #dtypes
        f.write('-Types-----------------------\n')
        f.write(data.dtypes.to_string())
        f.write('\n\n')

        #Shapes
        f.write('-Shapes-----------------------\n')
        f.write(str(data.shape))
        f.write('\n\n')

        #Nulls
        f.write('-Nulls-----------------------\n')
        f.write(data.isna().sum().to_string())
        f.write('\n\n')


def EncodingTrain(data, test = False):
    if test == False:
        categoricals = ['instkind','OC','sido','ownerChange']
        numericals = ['NCLiabilities',
        'debt',
        'liquidAsset',
        'liquidLiabilities',
        'longLoan',
        'netAsset',
        'nonCAsset',
        'quickAsset',
        'revenue',
        'salary',
        'sga',
        'shortLoan',
        'surplus',
        'tanAsset']
        
        # numericals = ['revenue', 'salescost', 'sga', 'salary', 'noi', 'noe',
        # 'interest', 'ctax', 'profit', 'liquidAsset', 'quickAsset',
        # 'receivableS', 'inventoryAsset', 'nonCAsset', 'tanAsset',
        # 'OnonCAsset', 'receivableL', 'debt', 'liquidLiabilities',
        # 'shortLoan', 'NCLiabilities', 'longLoan', 'netAsset', 'surplus',
        # 'employee']



        # OC 칼럼에 공백존재 ==>> 제거
        data['OC'] = data['OC'].str.strip()

        # 카테고리 변수를 변환
        for col in categoricals:
            data[col] = data[col].astype('category').cat.codes

        # KNN 알고리즘으로 채워 넣을 값을 정한다.
        imputer = KNNImputer(n_neighbors=5)
        # 다시 데이터프레임으로 만들어주고
        data_filled = pd.DataFrame(imputer.fit_transform(data))
        # 이전 칼럼명을 재설정
        data_filled.columns = data.columns

        # 카테고리 변수를 담을 데이터프레임
        categoric_data = pd.DataFrame()
        for col in categoricals:
            categoric_data[col] = data_filled[col].astype('category').cat.codes

        # 수치형 변수를 담을 데이터프레임
        numeric_data = pd.DataFrame()
        for col in numericals:
            numeric_data[col+'_diff'] = data_filled[col+'1'] - data_filled[col+'2']

        # MinMax-Scaling
        # 수치형 데이터만 MinMax 스케일링 적용
        stdScaler = StandardScaler()
        Scaled = stdScaler.fit_transform(numeric_data)


        # 수치형 데이터와 카테고리 데이터를 np.c_로 합치고 DataFrame으로 전환
        X  = pd.DataFrame(np.c_[np.array(categoric_data.drop(['OC','sido'],axis=1)),Scaled])
        # 칼럼명을 기존 칼럼명으로 바꾼다.
        X.columns = ['instkind','ownerChange','NCLiabilities',
                    'debt',
                    'liquidAsset',
                    'liquidLiabilities',
                    'longLoan',
                    'netAsset',
                    'nonCAsset',
                    'quickAsset',
                    'revenue',
                    'salary',
                    'sga',
                    'shortLoan',
                    'surplus',
                    'tanAsset']
        # X.columns = ['instkind','sido','ownerChange','revenue', 'salescost', 'sga', 'salary', 'noi', 'noe',
        # 'interest', 'ctax', 'profit', 'liquidAsset', 'quickAsset',
        # 'receivableS', 'inventoryAsset', 'nonCAsset', 'tanAsset',
        # 'OnonCAsset', 'receivableL', 'debt', 'liquidLiabilities',
        # 'shortLoan', 'NCLiabilities', 'longLoan', 'netAsset', 'surplus',
        # 'employee']
        y = data_filled.OC.astype('category').cat.codes
        y.columns = ['OC']
        return X,y
    else:
        data.drop('OC',axis=1,inplace=True)

        categoricals = ['instkind','sido','ownerChange']

        numericals = ['NCLiabilities',
        'debt',
        'liquidAsset',
        'liquidLiabilities',
        'longLoan',
        'netAsset',
        'nonCAsset',
        'quickAsset',
        'revenue',
        'salary',
        'sga',
        'shortLoan',
        'surplus',
        'tanAsset']


        # numericals = ['revenue', 'salescost', 'sga', 'salary', 'noi', 'noe',
        # 'interest', 'ctax', 'profit', 'liquidAsset', 'quickAsset',
        # 'receivableS', 'inventoryAsset', 'nonCAsset', 'tanAsset',
        # 'OnonCAsset', 'receivableL', 'debt', 'liquidLiabilities',
        # 'shortLoan', 'NCLiabilities', 'longLoan', 'netAsset', 'surplus',
        # 'employee']

        for col in categoricals:
            data[col] = data[col].astype('category').cat.codes

        # KNN 알고리즘으로 채워 넣을 값을 정한다.
        imputer = KNNImputer(n_neighbors=5)
        # 다시 데이터프레임으로 만들어주고
        data_filled = pd.DataFrame(imputer.fit_transform(data))
        # 이전 칼럼명을 재설정
        data_filled.columns = data.columns

        # 카테고리 변수를 담을 데이터프레임
        categoric_data = pd.DataFrame()
        for col in categoricals:
            categoric_data[col] = data_filled[col].astype('category').cat.codes

        # 수치형 변수를 담을 데이터프레임
        numeric_data = pd.DataFrame()
        for col in numericals:
            numeric_data[col+'_diff'] = data_filled[col+'1'] - data_filled[col+'2']

        # MinMax-Scaling
        # 수치형 데이터만 MinMax 스케일링 적용
        stdScaler = StandardScaler()
        Scaled = stdScaler.fit_transform(numeric_data)

        # 수치형 데이터와 카테고리 데이터를 np.c_로 합치고 DataFrame으로 전환
        X  = pd.DataFrame(np.c_[np.array(categoric_data.drop('sido',axis=1)),Scaled])
        # 칼럼명을 기존 칼럼명으로 바꾼다.
        X.columns = ['instkind','ownerChange','NCLiabilities',
                    'debt',
                    'liquidAsset',
                    'liquidLiabilities',
                    'longLoan',
                    'netAsset',
                    'nonCAsset',
                    'quickAsset',
                    'revenue',
                    'salary',
                    'sga',
                    'shortLoan',
                    'surplus',
                    'tanAsset']
        # X.columns = ['instkind','sido','ownerChange','revenue', 'salescost', 'sga', 'salary', 'noi', 'noe',
        # 'interest', 'ctax', 'profit', 'liquidAsset', 'quickAsset',
        # 'receivableS', 'inventoryAsset', 'nonCAsset', 'tanAsset',
        # 'OnonCAsset', 'receivableL', 'debt', 'liquidLiabilities',
        # 'shortLoan', 'NCLiabilities', 'longLoan', 'netAsset', 'surplus',
        # 'employee']

        return X


# X,y 데이터 입력시 모델 생성하고 학습 테스트 함수
def train_test(X,y):
    # 학습, 테스트 분리
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.2,random_state = 42)

    # XGboost 모델을 위한 파라미터 설정
    param = { 'n_estimators':100,
            'learning_rate':0.1,
            'gamma':1,
            'max_depth':10,
            'subsample':0.75,
            'colsample_bytree':1,
            'verbosity':0}

    # 학습할 모델 인스턴스 생성
    svc_model = SVC(kernel ='rbf',probability=True)
    dtc_model = DecisionTreeClassifier()
    rtc_model = RandomForestClassifier()
    xgb_model1 = xgb.XGBClassifier(**param)
    xgb_model2 = xgb.XGBClassifier(**param)
    ada_model = AdaBoostClassifier()
    gbc_model = GradientBoostingClassifier()
    rid_model = RidgeClassifier()
    

    # 모델 피팅
    svc_model.fit(Xtrain, ytrain)
    dtc_model.fit(Xtrain, ytrain)
    rtc_model.fit(Xtrain, ytrain)
    xgb_model1.fit(Xtrain, ytrain)
    ada_model.fit(Xtrain, ytrain)
    gbc_model.fit(Xtrain, ytrain)
    rid_model.fit(Xtrain, ytrain)

    # XGboost 모델을 위한 데이터 생성
    X_meta = np.c_[svc_model.predict(Xtrain),dtc_model.predict(Xtrain),rtc_model.predict(Xtrain),xgb_model1.predict(Xtrain),ada_model.predict(Xtrain),gbc_model.predict(Xtrain),rid_model.predict(Xtrain)]
    # XGboost 모델 학습
    xgb_model2.fit(X_meta,ytrain)
    
    # XGboost 모델을 위한 테스트 데이터 생성
    X_meta_test = np.c_[svc_model.predict(Xtest),dtc_model.predict(Xtest),rtc_model.predict(Xtest),xgb_model1.predict(Xtest),ada_model.predict(Xtest),gbc_model.predict(Xtest),rid_model.predict(Xtest)]

    # 모델 스코어 출력
    print('svc_model score : ', svc_model.score(Xtest, ytest))
    print('dtc_model score : ', dtc_model.score(Xtest, ytest))
    print('rtc_model score : ', rtc_model.score(Xtest, ytest))
    print('xgb_model1 score : ', xgb_model1.score(Xtest, ytest))
    print('xgb_model2 score : ', xgb_model2.score(X_meta_test,ytest))
    print('ada_model score : ', ada_model.score(Xtest, ytest))
    print('gbc_model score : ', gbc_model.score(Xtest, ytest))
    print('rid_model score : ', rid_model.score(Xtest, ytest))

    models = [svc_model, dtc_model, rtc_model,xgb_model1, ada_model, gbc_model, rid_model, xgb_model2]
    return models


# 예측 함수
def prediction(models,test_data):
    answer = pd.read_csv('data/submission_sample.csv')
    answer_list = []
    for md in models[:-1]:
        answer_list.append(md.predict(test_data))
    for ans in answer_list[1:]:
        answer_list[0] = np.c_[answer_list[0],ans]
    model = models[-1]
    answer_sheet = np.array(model.predict(answer_list[0]))
    answer['OC'] = answer_sheet
    answer.set_index('inst_id',inplace=True)

    return answer.to_csv('answer.csv',mode = 'w')

# 메인 구동함수
def main_func():
    warnings.filterwarnings(action='ignore')
    data = load_data('data/train.csv')
    analisys(data)
    X,y = EncodingTrain(data)
    analisys(pd.DataFrame(np.c_[X,y]),axis=1)
    print(X)
    print(y)
    models = train_test(X,y)

    test_data = load_data('data/test.csv')
    Xtest= EncodingTrain(test_data,test=True)
    prediction(models, Xtest)

# py 파일 실행시 작동되는 함수
main_func()
    



    
