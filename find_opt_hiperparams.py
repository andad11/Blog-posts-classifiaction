from model import tf_idf_transform, identity_tokenizer
from preprocess import df_blog, month_dummies_cols
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#%%
df = df_blog.copy()

# %%
targets = 'gender'
non_text_features = ['word_number_norm', 'mean_letters_per_word_norm'] + month_dummies_cols
features = 'token'

lgbm = lgb.LGBMClassifier(learning_rate=0.1)

#%%
X_train, X_test, y_train, y_test = train_test_split(df_blog[features], df_blog[targets], test_size=0.2,
                                                    random_state=42)

#%% Pipeline

pipe = Pipeline([('tfidf',TfidfVectorizer(lowercase=False)), ('nb', MultinomialNB())])
params = {
    'tfidf__max_df': (0.6, 0.7, 0.8),
    'tfidf__min_df': (5,10),
    'tfidf__max_features': (5000, 10000, 20000),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'nb__alpha': (0.1, 0.3, 0.5, 0.8, 1),
}



#%%
cv_search = RandomizedSearchCV(estimator=pipe, param_distributions = params, cv=5, scoring='accuracy', verbose=2,
                               n_iter=20)
fitted_model = cv_search.fit(X_train, y_train)

#%%
print(cv_search.best_score_)
print(cv_search.best_params_)

#%%
y_pred = cv_search.predict(X_test)

print(f'Mean accuracy score: {accuracy_score(y_test, y_pred):.3f}')
lgbm_params = {
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'n_estimators': [50, 100, 200,500],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [5, 10, 15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
    }

''' 
LGBM parameters
{'colsample_bytree': 0.7, 'max_depth': 5, 'min_data_in_leaf': 30, 'min_split_gain': 0.4, 'n_estimators': 500,
 'num_leaves': 50, 'reg_alpha': 1.1, 'reg_lambda': 1.3, 'subsample': 0.8, 'subsample_freq': 20}
'''

'''
NB params
{'nb__alpha': 0.1, 'tfidf__max_df': 0.5, 'tfidf__max_features': 20000}
'''