from keras_retinanet.bin.train import *


if __name__ == '__main__':
    args = [
        '--steps', '100',
        '--epochs', '30',
        '--snapshot-path', os.path.join('data', 'model_snapshots'),
        '--tensorboard-dir', os.path.join('data', 'tensorboard_logs'),
        'csv',
        os.path.join('data', 'training', 'annotations.csv'),
        os.path.join('data', 'training', 'classes.csv'),
        '--val-annotations', os.path.join('data', 'validation', 'annotations.csv'),
    ]
    main(args)
