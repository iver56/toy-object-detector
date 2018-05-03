import os

from keras_retinanet.bin.evaluate import main

if __name__ == '__main__':
    args = [
        'csv',
        os.path.join('data', 'test', 'annotations.csv'),
        os.path.join('data', 'test', 'classes.csv'),
        os.path.join('data', 'model_snapshots', 'resnet50_csv_03.h5'),
        '--score-threshold=0.5',
        '--max-detections=40',
        '--save-path={}'.format(os.path.join('data', 'test_result')),
        '--convert-model',
    ]
    main(args)
