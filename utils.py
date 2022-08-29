import tensorflow as tf
from configs import cfg

__all__ = ["get_dataset", "img_load", "vgg_preprocessing"]


def dataset_generator(dataset_img_list):
    for path in dataset_img_list:
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, size=cfg.input_size[:2]) / 255.
        yield img


def get_dataset(dataset_img_list):
    ds = tf.data.Dataset.from_generator(dataset_generator,
                                        tf.float32,
                                        args=(dataset_img_list, ))
    ds = ds.shuffle(buffer_size=5000).batch(cfg.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def img_load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, size=cfg.input_size[:2])
    img = tf.expand_dims(img, axis=0)
    img = tf.concat([img for _ in range(cfg.batch_size)], axis=0)
    return img


def vgg_preprocessing(x):
    return tf.keras.applications.vgg16.preprocess_input(x)

