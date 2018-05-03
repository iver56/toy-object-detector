import os

import numpy as np
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

if __name__ == '__main__':
    model = load_model(
        os.path.join('data', 'model_snapshots', 'resnet50_csv_30.h5'),
        convert=True,
        nms=True  # non-max suppression
    )

    #print(model.summary())

    idx_to_class_name = {
        0: 'circle',
        1: 'rectangle',
        2: 'triangle',
    }

    image = read_image_bgr(os.path.join('data', 'test', '00000.png'))

    image = preprocess_image(image)
    image, scale = resize_image(image)

    inputs = np.expand_dims(image, axis=0)

    boxes, scores, labels = model.predict_on_batch(inputs)

    boxes /= scale

    # iterate over detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        b = box.astype(int)

        caption = "{} {:.3f}: {}".format(idx_to_class_name[label], score, str(b))
        print(caption)
