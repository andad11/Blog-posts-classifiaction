from datetime import datetime
from nltk.tokenize import word_tokenize, TweetTokenizer
from import_data import df_blog
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MinMaxScaler
import string
from nltk.corpus import stopwords
import pandas as pd

#%% Parse dates
df_blog.date = df_blog.date.apply(lambda x: x.lower())
df_blog.date = df_blog.date.str.replace(',,', df_blog.iloc[0,:]['date'])


#%% translate months

month_dict = {'julio' : 'july', 'mai':'may', 'mars':'march', 'juin':'june', 'juillet' : 'july',
              'julho' : 'july', 'agosto':'august','junho':'june', 'junio':'june', 'juli':'july', 'abril':'april',
              'mayo' : 'may', 'septiembre':'september', 'januar,':'january,', 'februar,':'february,', 'juni':'june',
              'mei' : 'may', 'febrero' : 'february', 'marzo':'march', 'octubre':'october', 'giugno':'czerwiec',
              'luglio':'july', 'janvier':'january', 'avril':'april', 'septembre':'september', 'octobre':'october',
              'novembre':'november', 'augusti':'august', 'aprill':'april', 'augustus':'august', 'dezember':'december',
              'maj':'may', 'maio':'may', 'septembrie':'september', 'noiembrie':'november', 'ianuarie':'january',
              'februarie':'february', 'iulie':'july', 'avgust':'august', 'ottobre':'october', 'jaanuar':'january',
              'juuni':'june', 'juuli':'july', 'setembro':'september', 'novembro':'november', 'czerwiec':'june',
              'lipiec':'july', 'kolovoz':'august', 'lipanj':'july', 'noviembre':'november', 'diciembre':'december',
              'enero' : 'january', 'outubro':'october', 'dezembro':'december', 'janeiro':'january',
              'fevereiro':'february', 'desember':'december', 'toukokuu' : 'may', 'elokuu':'august',
              'maart': 'march' }

for key, value in month_dict.items():
    df_blog.date = df_blog.date.str.replace(key, value)

#%%
df_blog['month'] = df_blog.date.apply(lambda x: x.split(',')[1])
df_blog['date_dformat'] = df_blog.date.apply(lambda x: datetime.strptime(x, '%d,%B,%Y'))
month_dummies = pd.get_dummies(df_blog.month, drop_first=True)
month_dummies_cols = month_dummies.columns.to_list()
df_blog = pd.concat([df_blog, month_dummies], axis =1)

#%%
df_blog['contains_URL'] = df_blog['text'].apply(lambda x: 'urlLink' in x)
df_blog['contains_signs'] = df_blog['text'].apply(lambda x: '&gt;' in x)

#%% Remove punctuation
punct =[]
punct += list(string.punctuation)

def remove_punctuations(text):
    for punctuation in punct:
        text = text.replace(punctuation, ' ')
    return text

#%% Stem and tokenize

def stem_and_tokenize(text):
    words = TweetTokenizer(reduce_len=True, strip_handles=True).tokenize(text)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    if '!' in  words:
        words.remove('!')
    return words

#%% Clean text

df_blog['token'] = df_blog['text']

df_blog['token'] = df_blog['token'].apply(lambda x: x.lower())
df_blog['token'] = df_blog['token'].str.replace('http\S+|www.\S+', '', case=False)
df_blog['token'] = df_blog['token'].str.replace('urlLink', '')
df_blog['token'] = df_blog['token'].apply(remove_punctuations)
df_blog['token'] = df_blog['token'].apply(lambda x: str(x).replace(" s ", " "))
df_blog['token'] = df_blog['token'].apply(stem_and_tokenize)
df_blog['token'] = df_blog['token'].apply(lambda list_data: [x for x in list_data if x.isalpha()])



#%% Normalize continous features
def count_letters(word_list):
    sum = 0
    for word in word_list:
        sum += len(word)
    return sum

df_blog['word_number'] = df_blog['token'].apply(len)
df_blog['mean_letters_per_word'] = df_blog['token'].apply(count_letters)/df_blog['word_number']
df_blog['mean_letters_per_word'] = df_blog['mean_letters_per_word'].fillna(0)

scaler = MinMaxScaler()
df_blog['word_number_norm'] = scaler.fit_transform(df_blog.word_number.values.reshape(-1,1))
df_blog['mean_letters_per_word_norm'] = scaler.fit_transform(df_blog.mean_letters_per_word.values.reshape(-1,1))

#%% Fileter out posts with no information after preprocess
df_blog = df_blog[df_blog['word_number'] != 0]