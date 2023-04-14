#surprise의 SVD 라이브러리를 사용하여 코드 작성

import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import datetime

# Load data from CSV file using pandas
df = pd.read_csv('../data/train_ratings.csv')
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['user_id', 'isbn', 'rating']], reader)

# Use SVD algorithm with default hyperparameters
algo = SVD()

# Evaluate the performance of the algorithm using cross validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model on the entire dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Load test data from CSV file using pandas
test_df = pd.read_csv('./data/test_ratings.csv')

# Create a list of tuples in the format of (userId, movieId, rating) from the test data
testset = list(zip(test_df['user_id'].values, test_df['isbn'].values, test_df['rating'].values))

# Predict ratings for the test set
predictions = algo.test(testset)

# Save predictions to a CSV file with execution time in filename and record the end time of the program
end_time = datetime.datetime.now()

filename = './data/predictions' + str(end_time) + '.csv'
pred_df = pd.DataFrame([(pred.uid, pred.iid, pred.est) for pred in predictions], columns=['user_id', 'isbn', 'rating'])
pred_df.to_csv(filename, index=False)