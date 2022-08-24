import tensorflow as tf


def content_loss(target_feature, pred_feature):
    feature_shape = tf.shape(pred_feature)
    batch_size = feature_shape[0]
    scale_factor = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)

    target_feature = tf.reshape(target_feature, shape=(batch_size, -1))
    pred_feature = tf.reshape(pred_feature, shape=(batch_size, -1))
    return tf.norm(pred_feature - target_feature, axis=1) / scale_factor


def gram_matrix(feature_map):
    feature_shape = tf.shape(feature_map)
    scale_factor = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)

    gram = tf.einsum("bijc,bijd->bcd", feature_map, feature_map)
    return gram / scale_factor


def style_loss(target_style, pred_style):
    target_gram = gram_matrix(target_style)
    pred_gram = gram_matrix(pred_style)

    return tf.norm(pred_gram - target_gram, ord="fro", axis=(1, 2))


def variants_loss(x):
    img_shape = tf.shape(x)
    a = tf.square(
        x[:, : img_shape[1] - 1, : img_shape[2] - 1, :] - x[:, 1:, : img_shape[2] - 1, :]
    )
    b = tf.square(
        x[:, : img_shape[1] - 1, : img_shape[2] - 1, :] - x[:, : img_shape[1] - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25), axis=[1, 2, 3])


if __name__ == "__main__":
    temp1 = tf.random.normal(shape=(4, 32, 32, 512))
    temp2 = tf.random.normal(shape=(4, 32, 32, 512))
    temp3 = tf.random.normal(shape=(4, 256, 256, 3))
    result = variants_loss(temp3)
    print(result)
