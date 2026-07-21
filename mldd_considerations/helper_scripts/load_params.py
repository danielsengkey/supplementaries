feature_selection = {
    'variancethreshold__threshold': [(.8 * (1 - .8)), (.9 * (1 - .9))],
    'selectpercentile__percentile': [80, 90, 100]
}

hgb_param = {
    **feature_selection,
    'estimator__max_iter': [1000, 10000, 99999999],
    'estimator__max_depth': [None, 10, 20, 50],
    'estimator__min_samples_leaf': [8, 16, 32, 64],
    'estimator__l2_regularization': [0, 0.1, 0.01],
    'estimator__learning_rate': [0.01],
    'estimator__warm_start': [True],
    'estimator__early_stopping': [True],
    'estimator__n_iter_no_change': [100],
    'estimator__random_state': [22]
}

rf_param = {
    **feature_selection,
    'estimator__n_estimators': [100, 500, 1000],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__oob_score':[True],
    'estimator__warm_start': [True],
    'estimator__min_samples_split': [2, 3, 4, 8],
    'estimator__max_features': ["sqrt", "log2", None],
    'estimator__random_state': [22]
}

knn_param =  {
    **feature_selection,
    'estimator__n_neighbors': [3, 5, 7, 9],
    'estimator__weights': ['uniform', 'distance'],
    'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'estimator__leaf_size': [10, 20, 30, 40, 50],
    'estimator__p': [1, 2],
    'estimator__n_jobs': [-1]
}

svc_param =  {
    **feature_selection,
    'estimator__C': [1, 10, 100, 200, 300],
    'estimator__degree': [2, 3, 4],
    'estimator__gamma': [0.001, 0.01, 0.005, 1],
    'estimator__coef0': [0.0, 0.1, 0.5],
    'estimator__shrinking': [True, False],
    'estimator__random_state': [22]
}

svr_param =  {
    'variancethreshold__threshold': [(.8 * (1 - .8)), (.9 * (1 - .9))],
    'selectpercentile__percentile': [50, 80, 90, 100],
    'estimator__C': [1, 10, 100, 200, 300],
    'estimator__degree': [2, 3, 4],
    'estimator__gamma': [0.001, 0.01, 0.005, 1],
    'estimator__coef0': [0.0, 0.1, 0.5],
    'estimator__shrinking': [True, False]
}

dt_param =  {
    **feature_selection,
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__min_samples_split': [2, 3, 4, 8],
    'estimator__max_features': ["sqrt", "log2", None],
    'estimator__random_state': [22]
}