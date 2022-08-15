from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    training_grp = parser.add_argument_group('Training')
    training_grp.add_argument(
        '--validation_size', type=float, default=0.1,
    )
    training_grp.add_argument(
        '--cv_folds', type=int, default=5,
    )
    training_grp.add_argument(
        '--n_neighbors', type=int, default=5,
    )
    training_grp.add_argument(
        '--group_by', type=str, default='product_code',
    )
    training_grp.add_argument(
        '--add_indicator', type=bool, default=True,
    )
    training_grp.add_argument(
        '--unknown_value', type=int, default=-1,
    )
    training_grp.add_argument(
        '--random_state', type=int, default=123,
    )

    inference_grp = parser.add_argument_group('Inference')


