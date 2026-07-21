def create_pipeline(estimator, problem_type='auto'):
    """
    Create a generic ML pipeline with interchangeable estimator
    
    Parameters:
    -----------
    estimator: sklearn estimator or one with similar calls. The final estimator (regressor or classifier)
    problem_type: str ('auto', 'regression', 'classification')
        Type of problem. If 'auto', tries to infer from estimator type
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_classif, mutual_info_regression
    
    # Auto-detect problem type if not specified
    if problem_type == 'auto':
        if hasattr(estimator, 'predict_proba') or hasattr(estimator, 'predict_log_proba'):
            problem_type = 'classification'
        else:
            # This is a simple heuristic - you might want to refine it
            problem_type = 'regression'
    
    # Choose an appropriate feature selection method
    if problem_type == 'classification':
        selector = SelectPercentile(mutual_info_classif)
    else:
        selector = SelectPercentile(mutual_info_regression)
    
    return Pipeline([
        ("variancethreshold", VarianceThreshold()),
        ("selectpercentile", selector),
        ("estimator", estimator)
    ])
    

def fit_estimator_pipeline(
        algorithm: str,
        n_splits = 3,
        scoring = None,
        random_state = 22,
        problem_type='regression',
        x_train=None,
        y_train=None,
        x_test = None,
        y_test = None
    ):
    """
    Setting and running hyperparameter tuning for pipeline of various algorithms and supervised Machine Learning tasks. This function returns the fitted GridSearchCV object of the pipeline.

    Parameters:
    algorithm: name of algorithm to be used. It is one of rf, hgbr, svm, dt, knn.
    target_name: a string of the combination of species and the drug target. It should be one of hcv_ns3, hcv_ns34a, hcv_ns5a, hcv_ns5b, homosapiens_bace1, homosapiens_cyp17a1, sars-cov-2_pp1ab.
    n_splits: the number of folds for the cross-validation.
    scoring: evaluation metric(s) to be used in GridSearchCV
    random_state: random seed for reproducibility.
    problem_type: 'regression' or 'classification'
    x_train: training features
    y_train: training labels
    """

    valid_algorithms = ["rf", "hgb", "svm", "dt", "knn"]

    if algorithm not in valid_algorithms:
        raise ValueError("algorithm has to be one of rf, hgb, svm, dt, knn")
    if problem_type not in ['regression', 'classification']:
        raise ValueError("problem_type has to be either 'regression' or 'classification'")
    if scoring is None:
        scoring = "r2" if problem_type == 'regression' else 'accuracy'

    # Setting up pipeline and hyperparameter grid based on the selected algorithm
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold
    
    from sklearn.metrics import r2_score, accuracy_score
    
    print(f"Fitting estimator pipeline for {algorithm} with {problem_type} task.")

    match algorithm:                
        case "rf":
            if problem_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                estimator_pipe = create_pipeline(RandomForestRegressor(random_state=random_state), problem_type= problem_type)
            else:
                from sklearn.ensemble import RandomForestClassifier
                estimator_pipe = create_pipeline(RandomForestClassifier(random_state=random_state), problem_type=problem_type)
            
            from ml_libs.load_params import rf_param 
            estimator_params = rf_param

        case "hgb":
            if problem_type == 'regression':
                from sklearn.ensemble import HistGradientBoostingRegressor
                estimator_pipe = create_pipeline(HistGradientBoostingRegressor(random_state=random_state), problem_type=problem_type)
                
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                estimator_pipe = create_pipeline(HistGradientBoostingClassifier(random_state=random_state), problem_type=problem_type)
                
            from ml_libs.load_params import hgb_param
            estimator_params = hgb_param
            
        case "svm":
            if problem_type == 'regression':
                from sklearn.svm import SVR
                from ml_libs.load_params import svr_param
                
                estimator_pipe = create_pipeline(SVR(), problem_type=problem_type)
                estimator_params = svr_param

            else:
                from sklearn.svm import SVC
                from ml_libs.load_params import svc_param
                
                estimator_pipe = create_pipeline(SVC(), problem_type=problem_type)
                estimator_params = svc_param         

        case "dt":
            if problem_type == 'regression':
                from sklearn.tree import DecisionTreeRegressor
                estimator_pipe = create_pipeline(DecisionTreeRegressor(random_state=random_state), problem_type=problem_type)
                
            else:
                from sklearn.tree import DecisionTreeClassifier
                estimator_pipe = create_pipeline(DecisionTreeClassifier(random_state=random_state), problem_type=problem_type)

            from ml_libs.load_params import dt_param
            estimator_params = dt_param

        case "knn":
            if problem_type == 'regression':
                from sklearn.neighbors import KNeighborsRegressor
                estimator_pipe = create_pipeline(KNeighborsRegressor(), problem_type=problem_type)
                
            else:
                from sklearn.neighbors import KNeighborsClassifier
                estimator_pipe = create_pipeline(KNeighborsClassifier(), problem_type=problem_type)

            from ml_libs.load_params import knn_param
            estimator_params = knn_param

        case _:
            raise ValueError("Unsupported algorithm in case statement")
    
    # Load the dataset and then fit the estimator grid
    estimator_grid = GridSearchCV(
        estimator_pipe,
        estimator_params,
        verbose = 2,
        n_jobs=-1,
        pre_dispatch = '1*n_jobs',
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring = scoring,
        return_train_score=True,
        error_score="raise"
    )
    estimator_grid.fit(
        X = x_train,
        y = y_train
    )

    return estimator_grid

def call_fit_then_save(
    algorithm: str,
    n_splits = 3,
    random_state = 22,
    problem_type='regression',
    x_train=None,
    y_train=None,
    x_test = None,
    y_test = None,
    model_dir = "models/",
    session_dir = "sessions/",
    fn_pattern = "model.pkl"
):
    valid_algorithms = ["rf", "hgb", "svm", "dt", "knn"]
    valid_tasks = ["regression", "classification"]

    if algorithm not in valid_algorithms:
        raise ValueError("algorithm has to be one of rf, hgb, svm, dt, knn")
    if problem_type not in valid_tasks:
        raise ValueError("problem_type has to be either 'regression' or 'classification'")      
    
    import dill
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    # Fit model
    print(f"Starting to fit and save model {fn_pattern}")
    fitted_pipeline = fit_estimator_pipeline(
        algorithm = algorithm,
        n_splits = n_splits,
        scoring = None,
        random_state = random_state,
        problem_type = problem_type,
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test
    )

    print("Best score: ", fitted_pipeline.best_score_)
    
    # Save model
    model_path = f"{model_dir}{fn_pattern}"
    session_path = f"{session_dir}{fn_pattern.replace('.pkl', '.session')}"

    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        dill.dump(fitted_pipeline, f)

    print(f"Saving session to {session_path}")
    dill.dump_session(session_path)

def model_training(
    target_name: str,
    n_splits = 3,
    random_state = 22
):
    """
    Define pipeline and run model training for various algorithms and supervised Machine Learning tasks for a particular target.

    Parameters:
    target_name: a string of the combination of species and the drug target. It should be one of hcv_ns3, hcv_ns34a, hcv_ns5a, hcv_ns5b, homosapiens_bace1, homosapiens_cyp17a1, sars-cov-2_pp1ab.
    n_splits: the number of folds for the cross-validation.
    """

    from ml_libs.load_conventions import target_name as tn_list
    from ml_libs.load_conventions import FP_list

    from ml_libs.scaffold_aware_split import scaffold_aware_split

    # Input validation
    if target_name not in tn_list:
        raise ValueError("target_name should be one of hcv_ns3, hcv_ns34a, hcv_ns5a, hcv_ns5b, homosapiens_bace1, homosapiens_cyp17a1, sars-cov-2_pp1ab")
    
    valid_algorithms = ["rf", "hgb", "svm", "dt", "knn"]
    valid_tasks = ["regression", "classification"]
    murcko_split_strategy = ["random", "stratified", "strict"]

    # Directory to save models
    import os
    model_dir = "models/"
    session_dir = "sessions/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    from datetime import datetime
    now = datetime.now()
    formatted_datetime_string = now.strftime("%Y%m%d%H%M%S")

    gs_time_fn = f"GridSearchCV_Time-{formatted_datetime_string}.csv"
    
    import time # To record time needed for a gridsearchcv tuning
    # Loop over fingerprints, splitting strategies, algorithms, and problem types
    for fingerprint in FP_list:
        for strategy in murcko_split_strategy:
            for algorithm in valid_algorithms:
                for problem_type in valid_tasks:
                    print(f"Training {algorithm} for {target_name} using {fingerprint} with {strategy} split for {problem_type} task.")
                    
                    if problem_type == 'classification':
                        target_col = 'bioactivity_class'

                        for drop_intermediate in [True, False]:
                            print(f"Drop intermediate: {drop_intermediate}")

                            # Scaffold-aware split for classification
                            x_train, x_test, y_train, y_test = scaffold_aware_split(
                                target_name = target_name,
                                features_target_dir = "../dataset/features_target/",
                                fingerprint = fingerprint,
                                scaffold_col='murcko_fragment',
                                strategy=strategy,
                                test_size=0.2,
                                rare_threshold=2,
                                debug_dataset = False,
                                target_col = target_col,
                                drop_intermediate = drop_intermediate,
                                random_state=random_state
                            )
                            fn_pattern = f"{target_name}-{fingerprint}-{problem_type}-{strategy}-dropIntermediate-{drop_intermediate}-{algorithm}.pkl"

                            start_time = time.perf_counter()
                            # Call fit and save function
                            call_fit_then_save(
                                algorithm = algorithm,
                                n_splits = n_splits,
                                random_state = random_state,
                                problem_type = problem_type,
                                x_train = x_train,
                                y_train = y_train,
                                x_test = x_test,
                                y_test = y_test,
                                model_dir = model_dir,
                                session_dir = session_dir,
                                fn_pattern = fn_pattern
                            )
                            end_time = time.perf_counter()
                            gridsearch_time = end_time - start_time

                            gs_time_msg = f"{algorithm}, {target_name}, {fingerprint}, {strategy}, {problem_type}, {drop_intermediate}, {gridsearch_time}\n"
                            
                            try:
                                with open(gs_time_fn, 'a') as file:
                                    file.write(gs_time_msg)
                                print(f"Record {gs_time_msg} to {gs_time_fn}.")
                            except IOError as e:
                                print(f"An error occurred: {e}")
                            
                            print(f"Completed training {algorithm} for {target_name} using {fingerprint} with {strategy} split for {problem_type} task with drop_intermediate={drop_intermediate} in {gridsearch_time}.")
                        
                    else: # regression
                        target_col = 'pIC50'
                        
                        # Scaffold-aware split for regression
                        x_train, x_test, y_train, y_test = scaffold_aware_split(
                            target_name = target_name,
                            features_target_dir = "../dataset/features_target/",
                            fingerprint = fingerprint,
                            scaffold_col='murcko_fragment',
                            strategy=strategy,
                            test_size=0.2,
                            rare_threshold=2,
                            debug_dataset = False,
                            target_col = target_col,
                            drop_intermediate = False,
                            random_state=random_state
                        )
                        fn_pattern = f"{target_name}-{fingerprint}-{problem_type}-{strategy}-{algorithm}.pkl"

                        # Call fit and save function
                        start_time = time.perf_counter()
                        call_fit_then_save(
                            algorithm = algorithm,
                            n_splits = n_splits,
                            random_state = random_state,
                            problem_type = problem_type,
                            x_train = x_train,
                            y_train = y_train,
                            x_test = x_test,
                            y_test = y_test,
                            model_dir = model_dir,
                            session_dir = session_dir,
                            fn_pattern = fn_pattern
                        )
                        end_time = time.perf_counter()
                        gridsearch_time = end_time - start_time

                        gs_time_msg = f"{algorithm}, {target_name}, {fingerprint}, {strategy}, {problem_type}, False, {gridsearch_time}\n"
                            
                        try:
                            with open(gs_time_fn, 'a') as file:
                                file.write(gs_time_msg)
                            print(f"Record {gs_time_msg} to {gs_time_fn}.")
                        except IOError as e:
                            print(f"An error occurred: {e}")
                            
                        print(f"Completed training {algorithm} for {target_name} using {fingerprint} with {strategy} split for {problem_type} task in {gridsearch_time}.")