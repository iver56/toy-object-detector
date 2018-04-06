import os

import scipy.misc
from generate_shapes.generate_shapes import generate_shapes


def get_image_filename(i):
    return '{0:05d}.png'.format(i)


def save_images_and_labels(output_directory, images, labels):
    if os.path.exists(output_directory):
        assert os.path.isdir(output_directory)
    else:
        os.makedirs(output_directory)
    for number, image in enumerate(images):
        path = os.path.join(output_directory, get_image_filename(number))
        scipy.misc.imsave(path, image)

    class_names = set()
    annotations_file_path = os.path.join(output_directory, 'annotations.csv')
    with open(annotations_file_path, 'w') as annotations_file:
        lines = []

        for i, image_labels in enumerate(labels):
            image_filename = get_image_filename(i)

            for shape_label in image_labels:
                class_names.add(shape_label.category)
                line = '{image_filename},{x1},{y1},{x2},{y2},{class_name}\n'.format(
                    image_filename=image_filename,
                    x1=shape_label.x1,
                    y1=shape_label.y1,
                    x2=shape_label.x2,
                    y2=shape_label.y2,
                    class_name=shape_label.category
                )
                lines.append(line)

        annotations_file.writelines(lines)

    classes_file_path = os.path.join(output_directory, 'classes.csv')
    with open(classes_file_path, 'w') as classes_file:
        class_names_sorted = sorted(class_names)
        lines = [
            '{},{}\n'.format(class_name, i)
            for i, class_name in enumerate(class_names_sorted)
        ]
        classes_file.writelines(lines)


if __name__ == '__main__':
    for set_name, num_images in [('training', 300), ('test', 100)]:
        print('Generating {} set ({} images)'.format(set_name, num_images))
        images, labels = generate_shapes(
            number_of_images=num_images,
            width=224,
            height=224,
            max_shapes=19,
            min_dimension=12,
            max_dimension=42
        )

        print('Saving {} set ({} images)'.format(set_name, num_images))
        save_images_and_labels(
            os.path.join('data', set_name),
            images,
            labels
        )

    print('Done')
