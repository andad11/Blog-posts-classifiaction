from model import tf_idf_transform
from preprocess import df_blog, month_dummies_cols
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV

#%%
df = df_blog.copy()

# %%
targets = 'gender'
non_text_features = ['word_number_norm', 'mean_letters_per_word_norm'] + month_dummies_cols
features = ['token'] + non_text_features

lgbm = lgb.LGBMClassifier(learning_rate=0.1)

#%%
X_train, X_test, y_train, y_test = train_test_split(df_blog[features], df_blog[targets], test_size=0.2,
                                                    random_state=42)
X_train, X_test = tf_idf_transform(X_train, X_test, text_col='token')

#%%
param_grid = {
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

gsearch = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
fitted_model = gsearch.fit(X_train, y_train)

#%%
print(gsearch.best_score_)
print(gsearch.best_params_)