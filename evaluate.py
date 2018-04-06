from keras_retinanet.bin.evaluate import *


def parse_args(args):
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument(
        '--annotations', help='Path to CSV file containing annotations for evaluation.',
        default=os.path.join('data', 'test', 'annotations.csv')
    )
    csv_parser.add_argument(
        '--classes', help='Path to a CSV file containing class label mapping.',
        default=os.path.join('data', 'test', 'classes.csv')
    )

    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument(
        '--score-threshold', help='Threshold on score to filter detections with (defaults to 0.5).',
        default=0.5, type=float
    )
    parser.add_argument(
        '--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
        default=0.5, type=float
    )
    parser.add_argument(
        '--max-detections', help='Max Detections per image (defaults to 100).', default=100, type=int
    )
    parser.add_argument(
        '--save-path', help='Path for saving images with detections.',
        default=os.path.join('data', 'test_result')
    )

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # print model summary
    # print(model.summary())

    # start evaluation
    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )

    # print evaluation
    for label, average_precision in average_precisions.items():
        print(generator.label_to_name(label), '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    main()
