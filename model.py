from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from preprocess import df_blog
from statistics import mean
from preprocess import month_dummies_cols
import time


def identity_tokenizer(text):
    return text

def tf_idf_transform(X_train, X_test, text_col):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1, 2),
                                       max_features=10000,
                                       tokenizer=identity_tokenizer, lowercase=False)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train[text_col].values)
    tfidf_test = tfidf_vectorizer.transform(X_test[text_col].values)
    print(f'TF-IDF features: {tfidf_vectorizer.get_feature_names()}')

    non_text_features = X_train.columns.to_list()
    non_text_features.remove(text_col)

    # Merge TF-IDF features with rest of features
    X_train = np.concatenate([tfidf_train.toarray(), X_train[non_text_features].to_numpy()], axis=1)
    X_test = np.concatenate([tfidf_test.toarray(), X_test[non_text_features].to_numpy()], axis=1)

    return X_train, X_test

def make_prediction(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f'Time elapsed in seconds: {(end - start):.2f}')
    return wrapper

@measure_time
def cross_validate(df, X_cols, y_cols, model, n_splits):
    
    kfold = KFold(n_splits= n_splits, shuffle= True, random_state=0)
    accuracy_scores = []
    
    for train, test in kfold.split(df):
        X_train, y_train = df.loc[train, X_cols], df.loc[train, y_cols]
        X_test, y_test = df.loc[test, X_cols], df.loc[test, y_cols]
        X_train, X_test = tf_idf_transform(X_train, X_test, text_col='token')
        y_pred = make_prediction(model, X_train = X_train, y_train= y_train, X_test= X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    print(accuracy_scores)
    print(f'Mean accuracy score: {mean(accuracy_scores) :.2f}')
    
    return accuracy_scores

#%%
#if targets == 'gender':
    # cm = confusion_matrix(y_test, y_pred, labels=['male', 'female'])
    # print(cm)
    #
    # precision = cm[0,0]/ (cm[0,0] + cm[1,0])
    # print(f'Precision: {precision:.2f}')
    #
    # recall = cm[0,0]/ (cm[0,0] + cm[0,1])
    # print(f'Recall: {recall:.2f}')


if __name__ == '__main__':
    # %%
    df = df_blog.copy()

    # %%
    targets = 'topic'
    non_text_features = ['word_number_norm', 'mean_letters_per_word_norm'] + month_dummies_cols
    features = ['token'] + non_text_features

    nb = MultinomialNB(0.1)
    svc = SVC(kernel='linear')
    lgbm = lgb.LGBMClassifier(num_leaves=31, max_depth=8, learning_rate=0.1, n_estimators=100)

    cross_val_score = cross_validate(df=df_blog, X_cols=features, y_cols=targets, model=lgbm, n_splits=5)