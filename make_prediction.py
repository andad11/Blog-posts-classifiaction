from preprocess import df_blog, month_dummies_cols
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pipline_params import lgbm_tfidf_best_params, tf_idf_params
from pipline_params import xgb_params as model_params



#%%
df = df_blog.copy()

# %% Choose target and features
targets = 'gender'
num_features = ['word_number', 'mean_letters_per_word']
cat_features = ['month']
text_features = 'token'
features = num_features + cat_features
features.append(text_features)

X_train, X_test, y_train, y_test = train_test_split(df_blog[features], df_blog[targets], test_size=0.2,
                                                    random_state=42)
#%% Create Pipeline

num_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
cat_transformer = Pipeline(steps=[('scaler', OneHotEncoder())])
text_transformer = Pipeline(steps=[('tfidf',TfidfVectorizer(lowercase=False, ngram_range=(1,2),
                                                            max_df=0.8, min_df=10, max_features=20000))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('text', text_transformer, text_features)])


pipe = Pipeline([('preprocess',preprocessor), ('model', xgb.XGBClassifier(objective='binary:logistic' ))])

#%% Define params


new_keys = ['model__'+key for key in model_params.keys()]
model_params = dict(zip(new_keys, list(model_params.values())))

pipeline_params = {**tf_idf_params, **model_params}

#%% Fit Random Search
cv_search = RandomizedSearchCV(estimator=pipe, param_distributions = pipeline_params, cv=5, scoring='accuracy',
                               verbose=2, n_iter=30)
fitted_model = cv_search.fit(X_train, y_train)

#%%
print(cv_search.best_score_)
print(cv_search.best_params_)

#%%
y_pred = cv_search.predict(X_test)
print(f'Mean accuracy score: {accuracy_score(y_test, y_pred):.3f}')



'''
NB params
{'nb__alpha': 0.1, 'tfidf__max_df': 0.5, 'tfidf__max_features': 20000}
'''