MODEL_DIR = "./model/"
DATA_DIR = "./data/"
PRED_DIR = "./predictions/"
TRAIN_FILENAME = "train.csv"

# search_params.group_by = 'product_code'
CAT_VARS = ['attribute_0', 'attribute_1']
INT_ATTRIBUTES = ['attribute_2', 'attribute_3']  # could be categorical or ordinal
INT_MEASUREMENTS = ['measurement_0', 'measurement_1', 'measurement_2']  # could be categorical or ordinal
NUM_VARS = [
    'measurement_3',
    'measurement_4',
    'measurement_5',
    'measurement_6',
    'measurement_7',
    'measurement_8',
    'measurement_9',
    'measurement_10',
    'measurement_11',
    'measurement_12',
    'measurement_13',
    'measurement_14',
    'measurement_15',
    'measurement_16',
    'measurement_17',
]
LABEL = 'failure'

# These definitions can be adjusted (e.g. can treat INT_MEASUREMENTS as categorical)
ENCODE_VARS = CAT_VARS 
SCALE_VARS = NUM_VARS + INT_ATTRIBUTES + INT_MEASUREMENTS
IMPUTE_VARS = NUM_VARS + INT_ATTRIBUTES + INT_MEASUREMENTS 

# Fixed ML hyperparameters:
FIXED_HP = {
    'max_iter': 500,
}

# Parameters:
# encoding = dict(
#     ordinal_encoder__unknown_value = -1
# )
# imputing = dict(
#     knn_imputer__n_neighbors = 5,
#     knn_imputer__group_by = search_params.group_by,
#     knn_imputer__add_indicator = True
# )
