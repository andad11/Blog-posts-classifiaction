from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from preprocess import df_blog

#%%
df= df_blog.copy()

#%%
targets = 'topic'
non_text_features = ['word_number_norm', 'mean_letters_per_word_norm']
features =['token'] + non_text_features

y = df[targets]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Tfidf Vectorizer

def identity_tokenizer(text):
    return text

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, min_df = 5, ngram_range=(1,2),
                                   max_features=10000,
                                   tokenizer= identity_tokenizer, lowercase=False)

tfidf_train = tfidf_vectorizer.fit_transform(X_train['token'].values)
tfidf_test = tfidf_vectorizer.transform(X_test['token'].values)
print(f'TF-IDF features: {tfidf_vectorizer.get_feature_names()}')

#%% Merge TF-IDF features with rest of features
X_train = np.concatenate([tfidf_train.toarray(), X_train[non_text_features].to_numpy()], axis = 1)
X_test = np.concatenate([tfidf_test.toarray(), X_test[non_text_features].to_numpy()], axis = 1)

#%% Models prediction

def make_prediction(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

nb = MultinomialNB(0.1)
svc = SVC(kernel='linear')
lgbm = lgb.LGBMClassifier(num_leaves=31, max_depth=8, learning_rate=0.1, n_estimators=100)

y_pred = make_prediction(nb)

#%% Print metrics
score = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {score :.2f}')

#%%
if targets == 'gender':
    cm = confusion_matrix(y_test, y_pred, labels=['male', 'female'])
    print(cm)

    precision = cm[0,0]/ (cm[0,0] + cm[1,0])
    print(f'Precision: {precision:.2f}')

    recall = cm[0,0]/ (cm[0,0] + cm[0,1])
    print(f'Recall: {recall:.2f}')