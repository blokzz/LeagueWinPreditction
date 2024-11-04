import json
import pandas as pd

df = pd.read_csv('games.csv')

champ_data = json.load(open('champion_info.json'))

# team 1 champs

df['t1_champ1'] = df['t1_champ1id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t1_champ2'] = df['t1_champ2id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t1_champ3'] = df['t1_champ3id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t1_champ4'] = df['t1_champ4id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t1_champ5'] = df['t1_champ5id'].apply(lambda x: champ_data['data'][str(x)]['name'])

# team 2 champs

df['t2_champ1'] = df['t2_champ1id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t2_champ2'] = df['t2_champ2id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t2_champ3'] = df['t2_champ3id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t2_champ4'] = df['t2_champ4id'].apply(lambda x: champ_data['data'][str(x)]['name'])
df['t2_champ5'] = df['t2_champ5id'].apply(lambda x: champ_data['data'][str(x)]['name'])

df = df[['t1_champ1','t1_champ2','t1_champ3','t1_champ4','t1_champ5','t2_champ1','t2_champ2','t2_champ3','t2_champ4','t2_champ5','firstBlood','firstTower','winner']]

encodings1= [pd.get_dummies(df[col],prefix='t1') for col in ['t1_champ1','t1_champ2','t1_champ3','t1_champ4','t1_champ5']]

combined_df1 = sum(encodings1)

encodings2= [pd.get_dummies(df[col],prefix='t2') for col in ['t2_champ1','t2_champ2','t2_champ3','t2_champ4','t2_champ5']]

combined_df2 = sum(encodings2)

df = df.join(combined_df1).join(combined_df2)

df= df.drop(['t1_champ1','t1_champ2','t1_champ3','t1_champ4','t1_champ5','t2_champ1','t2_champ2','t2_champ3','t2_champ4','t2_champ5'],axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X , y  = df.drop('winner',axis=1) , df['winner']

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)

clf = RandomForestClassifier(n_jobs = -1)

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))