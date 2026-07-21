def scaffold_aware_split(target_name: str, features_target_dir: str, fingerprint: str, target_col: str, drop_intermediate = False,
                         scaffold_col='scaffold', strategy='strict', test_size=0.2,
                         rare_threshold=2, random_state=22, debug_dataset = False):
    """
    Unified function for scaffold-based dataset splitting with three strategies.
    
    Parameters:
    - target_name: name of the bioactivity target (abbr) in the dataset directory
    - scaffold_col: Column name for scaffold identifiers
    - strategy: 'random' | 'stratified' | 'strict'
    - test_size: Fraction for test set (default 0.2)
    - rare_threshold: Scaffolds with ≤ this count are considered rare
    - random_state: Random seed for reproducibility
    - features_target_dir: location of the dataset
    - fingerprint: fingerprint type to be loaded
    - target_col: column that contains the target variable (e.g., 'pIC50', 'standard_value', 'bioactvitiy_class')
    - drop_intermediate: Should the compounds with intermediate bioactivity be dropped?
    
    Returns:
    - X_train, X_test, Y_train, Y_test: Split DataFrames
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupShuffleSplit
    from collections import defaultdict

    # Features-target file
    ft_file = features_target_dir + target_name + "-FP-" + fingerprint + "-pic50.csv"
    
    # Read dataset from file
    print(f"Reading from {ft_file}")
    df = pd.read_csv(ft_file)

    # Intermediate bioactivity class
    if drop_intermediate:
        print("Dropping compounds with intermediate bioactivity...")
        df = df[df["bioactivity_class"]!="intermediate"]
    
    # Count rows with at least one NA or NaN
    rows_with_na = df.isna().any(axis=1).sum()

    if(rows_with_na > 0):
        print(f"Found {rows_with_na} with NA or NaN.")
        print(f"Dropping rows with NA...")
        df = df.dropna()
    
    # Step 1: Identify rare scaffolds (without regrouping)
    scaffold_counts = df[scaffold_col].value_counts()
    rare_scaffolds = scaffold_counts[scaffold_counts <= rare_threshold].index.tolist()
    
    # Step 2: Add rarity flag
    df['is_rare'] = df[scaffold_col].isin(rare_scaffolds)
    
    if strategy == 'random':
        # Pure random split (ignores scaffolds)
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
    elif strategy == 'stratified':
        # Create strata: frequent scaffolds, rare scaffolds, and NO_SCAFFOLD
        df['strata'] = np.where(
            df[scaffold_col] == 'NO_SCAFFOLD', 'NO_SCAFFOLD',
            np.where(df['is_rare'], 'RARE', df[scaffold_col])
        )
        
        # Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=random_state
        )
        train_idx, test_idx = next(sss.split(df, df['strata']))
        train_df = df.iloc[train_idx].drop(['strata'], axis = 1)
        test_df = df.iloc[test_idx].drop(['strata'], axis = 1)
        
    elif strategy == 'strict':
        # For strict splitting, we need to handle three groups separately:
        # 1. Frequent scaffolds (group split)
        # 2. Rare scaffolds (distributed evenly)
        # 3. NO_SCAFFOLD (random split)
        
        # Split frequent scaffolds (no overlap)
        frequent_mask = (~df['is_rare']) & (df[scaffold_col] != 'NO_SCAFFOLD')
        df_frequent = df[frequent_mask]
        
        if len(df_frequent) > 0:
            gss = GroupShuffleSplit(
                n_splits=1, 
                test_size=test_size, 
                random_state=random_state
            )
            train_idx, test_idx = next(gss.split(
                df_frequent, 
                groups=df_frequent[scaffold_col]
            ))
            train_freq = df_frequent.iloc[train_idx]
            test_freq = df_frequent.iloc[test_idx]
        else:
            train_freq = pd.DataFrame()
            test_freq = pd.DataFrame()
        
        # Split rare scaffolds (distribute evenly)
        rare_mask = df['is_rare'] & (df[scaffold_col] != 'NO_SCAFFOLD')
        df_rare = df[rare_mask]
        
        train_rare, test_rare = [], []
        for scaffold in df_rare[scaffold_col].unique():
            scaffold_data = df_rare[df_rare[scaffold_col] == scaffold]
            if len(scaffold_data) == 1:
                train_rare.append(scaffold_data)  # Single example → train
            else:
                s_train, s_test = train_test_split(
                    scaffold_data, 
                    test_size=test_size, 
                    random_state=random_state
                )
                train_rare.append(s_train)
                test_rare.append(s_test)
        
        # Split NO_SCAFFOLD (random)
        no_scaffold_mask = df[scaffold_col] == 'NO_SCAFFOLD'
        df_no = df[no_scaffold_mask]
        train_no, test_no = train_test_split(
            df_no, 
            test_size=test_size, 
            random_state=random_state
        ) if len(df_no) > 0 else (pd.DataFrame(), pd.DataFrame())
        
        # Combine all splits
        train_df = pd.concat([train_freq] + train_rare + [train_no])
        test_df = pd.concat([test_freq] + test_rare + [test_no])
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose 'random', 'stratified', or 'strict'")

    other_target_col = 'pIC50' if target_col=='bioactivity_class' else 'bioactivity_class'

    X_train = train_df.drop([target_col], axis = 1) if debug_dataset else train_df.drop(['molecule_chembl_id', target_col, 'canonical_smiles', scaffold_col, 'is_rare', other_target_col], axis = 1) 
    Y_train = train_df[target_col]
    
    X_test = test_df.drop([target_col], axis = 1) if debug_dataset else test_df.drop(['molecule_chembl_id', target_col, 'canonical_smiles', scaffold_col, 'is_rare', other_target_col], axis = 1)
    Y_test = test_df[target_col]
    
    return X_train, X_test, Y_train, Y_test