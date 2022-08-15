import os
import joblib
import pandas as pd
import config.config as cfg


def load_model(model_path):
    return joblib.load(model_path)


def make_predictions(x, model, save_path = None):
    preds = model.predict(x)
    preds = pd.Series(preds, index=x.index, name='failure')
    if save_path is not None:
        preds.to_csv(save_path, index=True, index_label='id')
    return preds


def predict_from_file(train_path = None, model_path = None, save_path = None):
    if train_path is None:
        train_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILENAME)
    if model_path is None:
        model_path = os.path.join(cfg.MODEL_DIR, 'model.pkl')

    # Make sure the file is not empty
    with open(train_path, "r") as f:
        assert f.readline() != "", "Empty file!"

    data = pd.read_csv(train_path, index_col='id')
    assert data.shape[0] > 0, "Empty file!"

    x = data.drop(cfg.LABEL, axis=1)
    model = load_model(model_path)
    return make_predictions(x, model, save_path)


def predict_from_input(input, model_path = None, save_path = None):
    if model_path is None:
        model_path = os.path.join(cfg.MODEL_DIR, 'model.pkl')

    # TODO: write input validation functions
    pass

    if cfg.LABEL in input.columns:
        input.drop(cfg.LABEL, axis=1, inplace=True)  # drop label if exists
    return make_predictions(input, load_model(model_path), save_path)