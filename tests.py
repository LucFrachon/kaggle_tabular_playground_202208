import unittest
import os
from argparse import Namespace
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import hp_optimization as hp
import predict as pr
from config import config as cfg


cfg.CAT_VARS = ['catvar1', 'catvar2']
cfg.INT_ATTRIBUTES = ['intattr1', 'intattr2']
cfg.INT_MEASUREMENTS = ['intmeas1', 'intmeas2']
cfg.NUM_VARS = ['numvar1', 'numvar2']

# Make a test dataframe
n_sample = 100
x = pd.DataFrame(
    np.empty((n_sample, len(['grpvar'] + cfg.CAT_VARS + cfg.INT_ATTRIBUTES + cfg.INT_MEASUREMENTS + cfg.NUM_VARS))),
    columns=['grpvar'] + cfg.CAT_VARS + cfg.INT_ATTRIBUTES + cfg.INT_MEASUREMENTS + cfg.NUM_VARS
)
x['grpvar'] = np.random.choice(['A', 'B'], size=x.shape[0], replace=True)
x['catvar1'] = np.random.choice(['cat0', 'cat1'], size=x.shape[0], replace=True)
x['catvar2'] = np.random.choice(['cat1', 'cat2'], size=x.shape[0], replace=True)
x[
    cfg.INT_ATTRIBUTES + cfg.INT_MEASUREMENTS
] = np.random.randint(0, n_sample, size=(x.shape[0], len(cfg.INT_ATTRIBUTES + cfg.INT_MEASUREMENTS)))
x[cfg.NUM_VARS] = (
        np.random.randn(x.shape[0], len(cfg.NUM_VARS)) * np.random.randint(1, 10, size=(1, len(cfg.NUM_VARS)))
        + np.random.randint(5, 80, size=(1, len(cfg.NUM_VARS)))
)

# Add some NA values to the dataframe
x.loc[np.random.randint(0, len(x), size=10), 'intmeas1'] = np.nan
x.loc[np.random.randint(0, len(x), size=20), 'numvar1'] = np.nan
x.loc[np.random.randint(0, len(x), size=10), 'catvar1'] = np.nan

# Create fake labels
y = pd.Series(np.random.choice([0, 1], size=x.shape[0], replace=True), name=cfg.LABEL)

# Create dummy CLI parameters
cli_args = Namespace(
    test_size=0.1,
    cv_folds=5,
    scoring='roc_auc',
    random_state=123,
    model_filename='model.pkl',
    n_neighbors=5,
    group_by='grpvar',
    add_indicator=True,
    unknown_value=-1,
)

# @unittest.skip('Temporarily skipped')
class TestHPOptimization(unittest.TestCase):

    def test_make_preprocessing_pipeline(self):
        # Create a preprocessing pipeline
        pp_pipe = hp.make_preprocessing_pipeline(cli_args, cat_vars=['catvar1', 'catvar2'])
        x_out = pp_pipe.fit_transform(x)
        base_cols = [
                'catvar1_cat0', 'catvar1_cat1', 'catvar1_nan', 'catvar2_cat1', 'catvar2_cat2',
                'intattr1', 'intattr2', 'intmeas1', 'intmeas2', 'numvar1', 'numvar2',
        ]
        na_cols = ['missingindicator_intmeas1', 'missingindicator_numvar1']
        grp_cols = ['grpvar']

        if cli_args.add_indicator:
            self.assertTrue(set(x_out.columns) == set(base_cols + na_cols))
        else:
            self.assertTrue(set(x_out.columns) == set(base_cols))

        self.assertEqual(pp_pipe.steps[0][0], 'pp_ops')
        self.assertEqual(pp_pipe.steps[1][0], 'knn_impute')
        self.assertEqual(pp_pipe.steps[2][0], 'remove_grp_var')
        self.assertEqual(
            set(pp_pipe.steps[0][1].get_feature_names_out()), set(base_cols + grp_cols))

        if cli_args.add_indicator:
            self.assertEqual(
                set(pp_pipe.steps[1][1].get_feature_names_out()),
                set(base_cols + na_cols + grp_cols)
            )
            self.assertEqual(
                set(pp_pipe.steps[2][1].get_feature_names_out()),
                set(base_cols + na_cols)
            )
        else:
            self.assertEqual(
                set(pp_pipe.steps[1][1].get_feature_names_out()),
                set(base_cols + grp_cols)
            )
            self.assertEqual(
                set(pp_pipe.steps[2][1].get_feature_names_out()), set(base_cols)
            )
        self.assertEqual(set(x_out.columns), set(pp_pipe.steps[2][1].get_feature_names_out()))
        self.assertTrue(cli_args.group_by not in x_out.columns)
        self.assertTrue(x_out.isnull().sum().sum() == 0)
        for c in cfg.NUM_VARS:
            self.assertAlmostEqual(x_out[c].mean(), 0, delta=0.1)
            self.assertAlmostEqual(x_out[c].std(), 1, delta=0.1)
        self.assertTrue(all([x_out[c].dtype in [np.float64, np.int32, np.int64] for c in x_out.columns]))

    def test_make_ml_pipeline(self):
        pp_pipe = hp.make_preprocessing_pipeline(cli_args, cat_vars=['catvar1', 'catvar2'])
        ml_pipe = hp.make_ml_pipeline(
            pp_pipe,
            LogisticRegression,
            cfg.FIXED_HP,
            cli_args.random_state
        )
        self.assertTrue(ml_pipe.steps[0][1].fit(x) is not None)
        ml_pipe.fit(x, y)
        self.assertIsInstance(ml_pipe.steps[0][1], Pipeline)
        self.assertIsInstance(ml_pipe.steps[1][1], LogisticRegression)
        self.assertTrue(ml_pipe.score(x, y) is not np.nan)

    def test_make_seach_pipeline(self):
        pp_pipe = hp.make_preprocessing_pipeline(cli_args, cat_vars=['catvar1', 'catvar2'])
        ml_pipe = hp.make_ml_pipeline(
            pp_pipe,
            LogisticRegression,
            cfg.FIXED_HP,
            cli_args.random_state
        )
        n_groups = len(x[cli_args.group_by].unique())
        search_pipe = hp.init_grid_search(ml_pipe, n_groups, cli_args)
        self.assertIsInstance(search_pipe.estimator, Pipeline)
        self.assertIsInstance(search_pipe.estimator.steps[1][1], LogisticRegression)

    def test_run_search(self):
        pp_pipe = hp.make_preprocessing_pipeline(cli_args, cat_vars=['catvar1', 'catvar2'])
        ml_pipe = hp.make_ml_pipeline(
            pp_pipe,
            LogisticRegression,
            cfg.FIXED_HP,
            cli_args.random_state
        )
        n_groups = len(x[cli_args.group_by].unique())
        search = hp.init_grid_search(ml_pipe, n_groups, cli_args)

        best_estimator, best_score = hp.run_search(
            x, y, x[cli_args.group_by], search, filename='test_model.pkl'
        )
        self.assertTrue(best_estimator is not None)
        self.assertTrue((best_score != np.nan) and (best_score > 0))


class TestPredict(unittest.TestCase):
    def test_load_model(self):
        try:
            model = pr.load_model(os.path.join(cfg.MODEL_DIR, 'test_model.pkl'))
        except FileNotFoundError:
            print(f"Model not found at {os.path.join(cfg.MODEL_DIR, 'test_model.pkl')}")
            return

        self.assertTrue(model is not None)
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.steps[1][1], LogisticRegression)

    def test_make_predictions(self):
        model = pr.load_model(os.path.join(cfg.MODEL_DIR, 'test_model.pkl'))
        preds = pr.make_predictions(
            x,
            model,
            save_path=os.path.join(cfg.PRED_DIR, 'test_preds.csv'),
        )
        self.assertTrue(preds is not None)
        self.assertIsInstance(preds, pd.Series)
        self.assertTrue(preds.shape == (x.shape[0],))
        self.assertTrue(x.shape[0] >= preds.sum() >= 0)

    def test_predict_from_file(self):
        # Multiple rows
        data = pd.concat([x, y], axis=1)
        data.to_csv(os.path.join(cfg.DATA_DIR, 'multiple_rows.csv'), index=True, index_label='id')
        preds = pr.predict_from_file(
            train_path=os.path.join(cfg.DATA_DIR, 'multiple_rows.csv'),
            model_path=os.path.join(cfg.MODEL_DIR, 'test_model.pkl'),
            save_path=os.path.join(cfg.PRED_DIR, 'multiple_preds.csv'),
        )
        self.assertTrue(preds is not None)
        self.assertIsInstance(preds, pd.Series)
        self.assertTrue(preds.shape == (x.shape[0],))
        self.assertTrue(x.shape[0] >= preds.sum() >= 0)

        # Edge case with only 1 row of data
        lines = [
            "id,catvar1,catvar2,intattr1,intattr2,intmeas1,intmeas2,numvar1,numvar2,grpvar,failure",
            "0,cat0,cat2,1,2,30,40,0.1,0.2,B,1"
        ]
        with open(cfg.DATA_DIR + "one_row.csv", "w") as f:
            f.writelines('\n'.join(lines))
        preds = pr.predict_from_file(
            train_path=os.path.join(cfg.DATA_DIR, 'one_row.csv'),
            model_path=os.path.join(cfg.MODEL_DIR, 'test_model.pkl'),
            save_path=os.path.join(cfg.PRED_DIR, 'one_pred.csv'),
        )
        self.assertTrue(preds is not None)
        self.assertIsInstance(preds, pd.Series)
        self.assertTrue(preds.shape == (1,))

        # Edge case with empty rows:
        line = [
            "id,catvar1,catvar2,intattr1,intattr2,intmeas1,intmeas2,numvar1,numvar2,grpvar,failure",
        ]
        with open(cfg.DATA_DIR + "no_row.csv", "w") as f:
            f.writelines(line)
        self.assertRaises(
            AssertionError,
            pr.predict_from_file,
            train_path=os.path.join(cfg.DATA_DIR, 'no_row.csv'),
            model_path=os.path.join(cfg.MODEL_DIR, 'test_model.pkl'),
            save_path=os.path.join(cfg.PRED_DIR, 'no_pred.csv'),
        )

        # Edge case with empty file:
        open(cfg.DATA_DIR + "empty.csv", "w").close()
        self.assertRaises(
            AssertionError,
            pr.predict_from_file,
            train_path=os.path.join(cfg.DATA_DIR, 'empty.csv'),
            model_path=os.path.join(cfg.MODEL_DIR, 'test_model.pkl'),
            save_path=os.path.join(cfg.PRED_DIR, 'empty_pred.csv'),
        )

    def test_predict_from_input(self):
        preds = pr.predict_from_input(
            x,
            model_path=os.path.join(cfg.MODEL_DIR, 'test_model.pkl'),
            save_path=os.path.join(cfg.PRED_DIR, 'multiple_preds.csv')
        )
        print(preds)
        self.assertTrue(preds is not None)
        self.assertIsInstance(preds, pd.Series)
        self.assertTrue(preds.shape == (x.shape[0],))
        self.assertTrue(x.shape[0] >= preds.sum() >= 0)

if __name__ == '__main__':
    unittest.main()
