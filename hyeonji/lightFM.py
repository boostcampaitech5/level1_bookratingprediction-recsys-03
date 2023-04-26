import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix
from sklearn import metrics

### 1. 데이터 eda
def prefix_language_dict_gen(books):
    language_dict = {}
    
    # 한자리 prefix
    for i in [0, 1, 2, 3, 4, 5, 7]:
        lan = books[books['isbn'].apply(lambda x: x[:1])==str(i)]['language'].value_counts().index[0]
        language_dict[str(i)] = lan
        
    # 두자리 prefix
    for i in range(80, 95):
        try:
            lan = books[books['isbn'].apply(lambda x: x[:2])==str(i)]['language'].value_counts().index[0]
            language_dict[str(i)] = lan
        except:
            pass
    
    # 세자리 prefix 
    for i in range(950, 990):
        try:
            lan = books[books['isbn'].apply(lambda x: x[:3])==str(i)]['language'].value_counts().index[0]
            language_dict[str(i)] = lan
        except:
            pass
        
    return language_dict


def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

# data load
data_path = '/opt/ml/data/'

users = pd.read_csv(data_path + 'users.csv')
books = pd.read_csv(data_path + 'books.csv')
train = pd.read_csv(data_path + 'train_ratings.csv')
test = pd.read_csv(data_path + 'test_ratings.csv')
sub = pd.read_csv(data_path + 'sample_submission.csv')

# user data preprocessing and indexing
users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '')
users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])


modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
location_list = []
for location in modify_location:
    try:
        right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
        location_list.append(right_location)
    except:
        pass

for location in location_list:
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
    
users = users.drop(['location'], axis=1)

users.loc[(users['location_city'].notna()) & (users['location_city'] == 'managua'), 'location_state'] = 'managua'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'managua') , 'location_country'] = 'nicaragua'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'aladinma'), 'location_state'] = 'imo state'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'aladinma'), 'location_country'] = 'nigeria'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'llanelli'), 'location_state'] = 'wales'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'llanelli'), 'location_country'] = 'unitedkingdom'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'victoria'), 'location_state'] = 'british columbia'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'victoria'), 'location_country'] = 'canada'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'washington'), 'location_state'] = 'dc'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'washington'), 'location_country'] = 'usa'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'milton'), 'location_state'] = 'massachusetts'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'milton'), 'location_country'] = 'usa'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'york'), 'location_state'] = 'north yorkshire'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'york'), 'location_country'] = 'england'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'chester'), 'location_state'] = 'cheshire'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'chester'), 'location_country'] = 'england'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'orleans'), 'location_state'] = 'ontario'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'orleans'), 'location_country'] = 'canada'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'kent'), 'location_state'] = 'kent'
users.loc[(users['location_city'].notna()) & (users['location_city'] == 'kent'), 'location_country'] = 'england'

users['age'] = users['age'].fillna(int(users['age'].mean()))
users['age'] = users['age'].apply(age_map)

loc_city2idx = {v:k for k,v in enumerate(users['location_city'].unique())}
loc_state2idx = {v:k for k,v in enumerate(users['location_state'].unique())}
loc_country2idx = {v:k for k,v in enumerate(users['location_country'].unique())}

users['location_city'] = users['location_city'].map(loc_city2idx)
users['location_state'] = users['location_state'].map(loc_state2idx)
users['location_country'] = users['location_country'].map(loc_country2idx)

# book data preprocessing and indexing
language_dict = prefix_language_dict_gen(books)

for prefix, language in language_dict.items():
    books.loc[(books['isbn'].str.startswith(prefix)) & (books['language'].isna()), 'language'] = language

books.loc[books.language.isna(), 'language'] = 'en'

category2idx = {v:k for k,v in enumerate(books['category'].unique())}
publisher2idx = {v:k for k,v in enumerate(books['publisher'].unique())}
language2idx = {v:k for k,v in enumerate(books['language'].unique())}
author2idx = {v:k for k,v in enumerate(books['book_author'].unique())}

books['category'] = books['category'].map(category2idx)
books['publisher'] = books['publisher'].map(publisher2idx)
books['language'] = books['language'].map(language2idx)
books['book_author'] = books['book_author'].map(author2idx)

# indexing
ids = users['user_id'].unique()
isbns = books['isbn'].unique()

idx2user = {idx:id for idx, id in enumerate(ids)}
idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

user2idx = {id:idx for idx, id in idx2user.items()}
isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

train['user_id'] = train['user_id'].map(user2idx)
sub['user_id'] = sub['user_id'].map(user2idx)
test['user_id'] = test['user_id'].map(user2idx)
users['user_id'] = users['user_id'].map(user2idx)

train['isbn'] = train['isbn'].map(isbn2idx)
sub['isbn'] = sub['isbn'].map(isbn2idx)
test['isbn'] = test['isbn'].map(isbn2idx)
books['isbn'] = books['isbn'].map(isbn2idx)

#create sparse matrix from train
shape = (len(user2idx), len(isbn2idx))

train_matrix = coo_matrix((train['rating'].values, (train['user_id'].astype(int), train['isbn'].astype(int))), shape = shape)
test_matrix = coo_matrix((test['rating'].values, (test['user_id'].astype(int), test['isbn'].astype(int))), shape = shape)

#create user_feature sparse matrix
user_features_source = [(users['user_id'][i],
                        [users['age'][i], users['location_city'][i], users['location_state'][i],
                         users['location_country'][i]]) for i in range(users.shape[0])]
#create book_feature sparse matrix
book_features_source = [(books['isbn'][i],
                        [books['category'][i], books['publisher'][i], books['language'][i],
                         books['book_author'][i]]) for i in range(books.shape[0])]

ratings = pd.concat([train, test]).reset_index(drop=True)

dataset = Dataset()
dataset.fit(users=users['user_id'].unique(),
            items=books['isbn'].unique(),
            user_features = users[users.columns[1:]].values.flatten(),
            item_features= books[books.columns[1:]].values.flatten()
            )

user_features = dataset.build_user_features(user_features_source)
book_features = dataset.build_item_features(book_features_source)

#model train and predict
model = LightFM()
model.fit(train_matrix, user_features = user_features, item_features = book_features, epochs=20)

# prediction = model.predict(test_user_id, test_book_id, user_features = user_features, book_features = book_features)