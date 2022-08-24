__all__ = ["Configs"]


class Configs:
    def __init__(self):
        self.input_size = (256, 256, 3)
        self.content_layer_name = "block2_conv2"
        self.style_layer_names = ["block1_conv2",
                                  "block2_conv2",
                                  "block3_conv3",
                                  "block4_conv3"]
        self.content_weights = 1e0
        self.style_weights = 1e5
        self.tv_weights = 1e-6
