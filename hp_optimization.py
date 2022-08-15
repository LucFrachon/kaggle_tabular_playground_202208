"""
Contains the hyperparameter optimisation code. Produces a trained model with the best-found
hyperparameters.
"""

import joblib, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pandas as pd

import preprocessors
from config import config as cfg


def make_preprocessing_pipeline(cli_params, cat_vars = cfg.CAT_VARS):
    return Pipeline(
        [
            (
                'pp_ops',
                preprocessors.PreprocessingOperations(
                    unknown_value=cli_params.unknown_value,
                    group_by=cli_params.group_by,
                    cat_vars=cat_vars,
                )
            ),
            (
                'knn_impute',
                preprocessors.GroupKNNImputerDF(
                    n_neighbors=cli_params.n_neighbors,
                    group_by=cli_params.group_by,
                    add_indicator=cli_params.add_indicator,
                )
            ),
            (
                'remove_grp_var',
                preprocessors.ColumnRemover(cli_params.group_by)
            )
        ],
    )


def make_ml_pipeline(preprocessing_pl, ml_model, fixed_model_params, random_state):
    return make_pipeline(
        preprocessing_pl,
        ml_model(**fixed_model_params, n_jobs=1, random_state=random_state),
    )


def init_grid_search(ml_pl, n_groups, cli_params):
    with open('config/parameter_grid.json', 'r') as f:
        grid = json.load(f)
    assert isinstance(grid, dict) and len(grid) > 0, "Empty parameter grid!"

    return GridSearchCV(
        ml_pl,
        param_grid=grid,
        # in GroupKFold, the number of fold cannot be higher than the number of groups:
        cv=GroupKFold(n_splits=min(cli_params.cv_folds, n_groups)),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=3,
        refit=True,
        error_score='raise',
    )


def run_search(x_train, y_train, groups, search, filename = None):
    search.fit(x_train, y_train.values.ravel(), groups=groups)

    if filename is not None:
        joblib.dump(search.best_estimator_, os.path.join(cfg.MODEL_DIR, filename))
    print(f"Hyperparameter search completed; best parameters:\n {search.best_params_}")

    return search.best_estimator_, search.best_score_


def main(cli_params):
    x = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILENAME), index_col='id')
    x, y = x.drop(cfg.LABEL, axis=1), x.pop(cfg.LABEL)

    x_train = x.sample(frac=0.9, random_state=cli_params.random_state)
    y_train = y.loc[x_train.index]
    x_valid = x.drop(x_train.index, axis=0)
    y_valid = y.loc[x_valid.index]
    groups = x_train[cli_params.group_by]
    n_groups = len(groups.unique())

    pp_pipeline = make_preprocessing_pipeline(cli_params, cfg.CAT_VARS)
    ml_pipeline = make_ml_pipeline(
        pp_pipeline,
        LogisticRegression,
        cfg.FIXED_HP,
        cli_params.random_state
    )
    search_pl = init_grid_search(ml_pipeline, n_groups, cli_params)

    best_model, best_score = run_search(
        x_train,
        y_train,
        groups,
        search_pl,
        filename=cli_params.model_filename
    )
    print(f"Best model score: {best_score}")
    train_acc = best_model.score(x_train, y_train)
    valid_acc = best_model.score(x_valid, y_valid)
    print(
        f"Weighted train accuracy: {train_acc : .5f}\n"
        f"Weighted validation accuracy: {valid_acc : .5f}"
    )
    train_auc = roc_auc_score(best_model.predict(x_train), y_train)
    valid_auc = roc_auc_score(best_model.predict(x_valid), y_valid)
    print(f"Train AUC: {train_auc : .5f}\nValidation AUC: {valid_auc : .5f}")

    
if __name__ == '__main__':
    from argparse import ArgumentParser
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = ArgumentParser()

    # CLI arguments for HP Search
    hp_search_grp = parser.add_argument_group('Hyper-parameter search')
    hp_search_grp.add_argument(
        '--test_size', type=float, default=0.1,
    )
    hp_search_grp.add_argument(
        '--cv_folds', type=int, default=5,
    )
    hp_search_grp.add_argument(
        '--scoring', type=str, default='roc_auc',
    )
    hp_search_grp.add_argument(
        '--random_state', type=int, default=123,
    )
    hp_search_grp.add_argument(
        '--model_filename', type=str, default='model.pkl',
    )

    # CLI arguments for preprocessing pipeline
    preprocessing_grp = parser.add_argument_group('Preprocessing')
    preprocessing_grp.add_argument(
        '--n_neighbors', type=int, default=5,
    )
    preprocessing_grp.add_argument(
        '--group_by', type=str, default='product_code',
    )
    preprocessing_grp.add_argument(
        '--add_indicator', type=int, default=1,
    )
    preprocessing_grp.add_argument(
        '--unknown_value', type=int, default=-1,
    )
    params = parser.parse_args()

    main(params)
