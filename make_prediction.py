import lightgbm as lgb
import xgboost as xgb
from pipline_params import lgbm_tfidf_best_params, tf_idf_params
from pipline_params import nb_params as model_params
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from preprocess import df_blog, month_dummies_cols

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


pipe = Pipeline([('preprocess',preprocessor), ('model', MultinomialNB())])

#%% Define params

new_keys = ['model__'+key for key in model_params.keys()]
model_params = dict(zip(new_keys, list(model_params.values())))

#pipeline_params = {**tf_idf_params, **model_params}

#%% Fit Random Search
cv_search = RandomizedSearchCV(estimator=pipe, param_distributions = model_params, cv=5, scoring='accuracy',
                               verbose=2, n_iter=3)
fitted_model = cv_search.fit(X_train, y_train)

#%%
print(cv_search.best_score_)
print(cv_search.best_params_)

#%%
y_pred = cv_search.predict(X_test)
print(f'Mean accuracy score: {accuracy_score(y_test, y_pred):.3f}')

if targets == 'gender':
    print(f"Precision female {precision_score(y_test, y_pred, pos_label='female'):.3f}")
    print(f"Recall female {recall_score(y_test, y_pred, pos_label='female'):.3f}")

    cm = confusion_matrix(y_test, y_pred, labels = ['female', 'male'])
    print("Confusion matrix:\n female male\n", cm)