import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose, Lambda, Cropping2D, Layer, Activation
from keras.models import Model, Sequential
from configs import cfg


def build_conv_block(out_channel, kernel_size, stride, activation, name):
    block = Sequential([
        Conv2D(filters=out_channel, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False),
        BatchNormalization(),
        Activation(activation)
    ], name=name)
    return block


def build_upsample_block(out_channel, kernel_size, stride, name):
    block = Sequential([
        Conv2DTranspose(filters=out_channel, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu")
    ], name=name)
    return block


class ResidualBlock(Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", use_bias=False)
        self.conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", use_bias=False)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu1 = Activation("relu")
        self.crop = Cropping2D(cropping=2)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        crop = self.crop(inputs)
        out = x + crop
        return out


def build_net():
    inputs = Input(shape=(None, None, 3))
    pad_inputs = Lambda(lambda img: tf.pad(img, [[0, 0], [40, 40], [40, 40], [0, 0]], mode="REFLECT")
                        , name="reflection_padding")(inputs)
    x = build_conv_block(32, 9, 1, "relu", "input_conv1")(pad_inputs)
    x = build_conv_block(64, 3, 2, "relu", "input_conv2")(x)
    x = build_conv_block(128, 3, 2, "relu", "input_conv3")(x)
    x = ResidualBlock()(x)
    x = ResidualBlock()(x)
    x = ResidualBlock()(x)
    x = ResidualBlock()(x)
    x = ResidualBlock()(x)
    x = build_upsample_block(64, 3, 2, "upsample_conv1")(x)
    x = build_upsample_block(32, 3, 2, "upsample_conv2")(x)
    out = build_conv_block(3, 9, 1, "sigmoid", "out_conv")(x)

    return Model(inputs, out)


def build_loss_net():
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
    vgg.trainable = False

    layer_names = cfg.style_layer_names + [cfg.content_layer_name]
    outputs = dict([(layer.name, layer.output) for layer in vgg.layers if layer.name in layer_names])
    return Model(vgg.inputs, outputs)


if __name__ == "__main__":
    model = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
    print(model.summary())
