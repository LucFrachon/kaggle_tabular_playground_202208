"""
Contains the hyperparameter optimisation code. Produces a trained model with the best-found
hyperparameters.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import config, preprocessors
from hp_optimization import make_preprocessing_pipeline







if __name__ == '__main__':
    logreg_pipeline = make_pipeline(
        pp_pipeline,
        LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=123,
        )
    )

    search = GridSearchCV(
        logreg_pipeline,
        param_grid={
            'logisticregression__C': [0.01,.1, 1, 10],
            'logisticregression__penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'logisticregression__fit_intercept': [True, False],
            'logisticregression__class_weight': ['balanced', None],
            'logisticregression__solver': ['newton-cg', 'sag', 'saga'],
            'logisticregression__l1_ratio': [0.1, .3, .5, .7 , .9]
        },
        cv=GroupKFold(n_splits=5),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=3,
        refit=False,
    )

    search.fit(x_train, y_train.values.ravel(), groups=x_train.product_code)
    print(f"Best parameters: {search.best_params_}")
    joblib.dump(search.best_estimator_, os.path.join(OUTPUT_PATH, "search_output.pkl"))