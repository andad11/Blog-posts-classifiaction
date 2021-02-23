lgbm_tfidf_best_params = {
  'model__subsample_freq': [20],
  'model__subsample': [0.9],
  'model__reg_lambda': [1.3],
  'model__reg_alpha': [1.2],
  'model__num_leaves': [50],
  'model__n_estimators': [1000, 1500, 2000, 3000],
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