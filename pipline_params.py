lgbm_tfidf_best_params = {
  'model__subsample_freq': [20],
  'model__subsample': [0.9],
  'model__reg_lambda': [1.3],
  'model__reg_alpha': [1.2],
  'model__num_leaves': [50],
  'model__n_estimators': [1000],
  'model__min_split_gain': [0.3],
  'model__min_data_in_leaf': [50],
  'model__max_depth': [20],
  'model__colsample_bytree': [0.8]}

lgbm_params = {
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'n_estimators': [100, 200, 500, 700],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [5, 10, 15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
    }

tf_idf_params = {
    'preprocess__text__tfidf__max_df': (0.7, 0.8),
    'preprocess__text__tfidf__min_df': (5,10),
    'preprocess__text__tfidf__max_features': (10000, 20000),
    'preprocess__text__tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
}

nb_params = {'alpha':[0.1, 0.5, 1]}

xgb_params = param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [10,15,20],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}

xgb_best_params = {
    'subsample': [0.8],
    'reg_lambda': [1.3],
    'reg_alpha': [1.1],
    'n_estimators': [400],
    'max_depth': [15],
    'colsample_bytree': [0.7]
}