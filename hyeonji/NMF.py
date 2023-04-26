#suprise library를 사용한 NMF 구현

import pandas as pd
import numpy as np
import time
from surprise import NMF, Dataset, Reader
from surprise.model_selection import cross_validate


data_path = '/opt/ml/data/'

train = pd.read_csv(data_path + 'train_ratings.csv')
test = pd.read_csv(data_path + 'test_ratings.csv')
sub = pd.read_csv(data_path + 'sample_submission.csv')

reader = Reader(rating_scale=(1, 10))
train_data = Dataset.load_from_df(train[['user_id', 'isbn', 'rating']], reader)

model = NMF(n_factors=20, n_epochs=150)
cross_validate(model, train_data, measures=['RMSE'], cv=5, verbose=True)

trainset = train_data.build_full_trainset()
model.fit(trainset)

testset = list(zip(test['user_id'].values, test['isbn'].values, test['rating'].values))
predict = model.test(testset)

#파일명 설정
now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')
filename = f'/opt/ml/data/submit/{save_time}_nmf.csv'

pred_df = pd.DataFrame([(pred.uid, pred.iid, pred.est) for pred in predict], columns=['user_id', 'isbn', 'rating'])
pred_df.to_csv(filename, index=False)