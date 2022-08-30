import tensorflow as tf
from glob import glob
import os
from configs import cfg
from utils import *
from loss.losses import *
from network import transformation_net
import argparse


parser = argparse.ArgumentParser(description='Neural Style Transfer script')
parser.add_argument('--style_img', required=True, type=str, help="style image path")
parser.add_argument('--dataset_dir', required=True, type=str, help="content images directory path")

args = parser.parse_args()


if __name__ == "__main__":
    style_img_path = args.style_img
    dataset_img_list = glob(os.path.join(args.dataset_dir, "*"))

    transfer_net = transformation_net.build_net()
    loss_net = transformation_net.build_loss_net()

    ds = get_dataset(dataset_img_list)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    style_img = style_img_load(style_img_path)
    pre_style = vgg_preprocessing(style_img)
    style_feature = loss_net(pre_style)
    for epoch in range(1, cfg.num_epochs + 1):
        for i, content_img in enumerate(ds):
            with tf.GradientTape() as tape:
                combine_img = transfer_net(content_img) * 255.

                pre_content = vgg_preprocessing(content_img * 255.)
                pre_combine = vgg_preprocessing(combine_img)

                combine_feature = loss_net(pre_combine)

                # content loss
                content_feature = loss_net(pre_content)[cfg.content_layer_name]
                cl = cfg.content_weights * content_loss(content_feature, combine_feature[cfg.content_layer_name])

                # style loss
                sls = []
                for name in cfg.style_layer_names:
                    sls.append(style_loss(style_feature[name], combine_feature[name]))
                sl = cfg.style_weights * tf.add_n(sls)

                # variation loss
                tv = cfg.tv_weights * variants_loss(combine_img)

                batch_loss = tf.reduce_mean(cl + sl + tv)
            gradients = tape.gradient(batch_loss, transfer_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transfer_net.trainable_variables))
            if (i + 1) % 100 == 0:
                transfer_net.save_weights("./ckpt/checkpoint")
                print("EPOCH {} ITER {} LOSS {}".format(epoch, i + 1, batch_loss))
                print("saving ckpt")
    style_file_name = os.path.basename(style_img_path).split('.')[0]
    save_path = os.path.join("./models/", style_file_name)
    transfer_net.save(save_path)

