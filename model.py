import os

import tensorflow as tf

import config
import download


def kld_loss(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true = y_true / (eps + sum_per_image)

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred = y_pred / (eps + sum_per_image)

    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss


class MSINET(tf.keras.Model):
    """The class representing the MSI-Net based on the VGG16 model. It
       implements a definition of the computational graph using Keras layers,
       as well as functions related to network training and inference.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if config.PARAMS["device"] == "gpu":
            self._data_format = "channels_first"
            self._channel_axis = 1
            self._dims_axis = (2, 3)
        elif config.PARAMS["device"] in ("cpu", "tpu"):
            # TPU requires channels_last format (same as CPU)
            self._data_format = "channels_last"
            self._channel_axis = 3
            self._dims_axis = (1, 2)

        # Encoder layers (VGG16-based)
        self.conv1_1 = tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv1_conv1_1")
        self.conv1_2 = tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv1_conv1_2")
        self.pool1 = tf.keras.layers.MaxPooling2D(
            2, 2, data_format=self._data_format)

        self.conv2_1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv2_conv2_1")
        self.conv2_2 = tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv2_conv2_2")
        self.pool2 = tf.keras.layers.MaxPooling2D(
            2, 2, data_format=self._data_format)

        self.conv3_1 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv3_conv3_1")
        self.conv3_2 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv3_conv3_2")
        self.conv3_3 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv3_conv3_3")
        self.pool3 = tf.keras.layers.MaxPooling2D(
            2, 2, data_format=self._data_format)

        self.conv4_1 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv4_conv4_1")
        self.conv4_2 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv4_conv4_2")
        self.conv4_3 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu",
            data_format=self._data_format, name="conv4_conv4_3")
        self.pool4 = tf.keras.layers.MaxPooling2D(
            2, 1, padding="same", data_format=self._data_format)

        self.conv5_1 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu", dilation_rate=2,
            data_format=self._data_format, name="conv5_conv5_1")
        self.conv5_2 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu", dilation_rate=2,
            data_format=self._data_format, name="conv5_conv5_2")
        self.conv5_3 = tf.keras.layers.Conv2D(
            512, 3, padding="same", activation="relu", dilation_rate=2,
            data_format=self._data_format, name="conv5_conv5_3")
        self.pool5 = tf.keras.layers.MaxPooling2D(
            2, 1, padding="same", data_format=self._data_format)

        # ASPP layers
        self.aspp_conv1 = tf.keras.layers.Conv2D(
            256, 1, padding="same", activation="relu",
            data_format=self._data_format, name="aspp_conv1_1")
        self.aspp_conv2 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu", dilation_rate=4,
            data_format=self._data_format, name="aspp_conv1_2")
        self.aspp_conv3 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu", dilation_rate=8,
            data_format=self._data_format, name="aspp_conv1_3")
        self.aspp_conv4 = tf.keras.layers.Conv2D(
            256, 3, padding="same", activation="relu", dilation_rate=12,
            data_format=self._data_format, name="aspp_conv1_4")
        self.aspp_conv5 = tf.keras.layers.Conv2D(
            256, 1, padding="valid", activation="relu",
            data_format=self._data_format, name="aspp_conv1_5")
        self.aspp_conv_out = tf.keras.layers.Conv2D(
            256, 1, padding="same", activation="relu",
            data_format=self._data_format, name="aspp_conv2")

        # Decoder layers
        self.decoder_conv1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="relu",
            data_format=self._data_format, name="decoder_conv1")
        self.decoder_conv2 = tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="relu",
            data_format=self._data_format, name="decoder_conv2")
        self.decoder_conv3 = tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="relu",
            data_format=self._data_format, name="decoder_conv3")
        self.decoder_conv4 = tf.keras.layers.Conv2D(
            1, 3, padding="same",
            data_format=self._data_format, name="decoder_conv4")

    def _upsample(self, stack, target_shape, factor):
        """This function resizes the input to a desired shape via the
           bilinear upsampling method.

        Args:
            stack (tensor, float32): A 4D tensor with the function input.
            target_shape (tensor, int32): A 1D tensor with the reference shape.
            factor (scalar, int): An integer denoting the upsampling factor.

        Returns:
            tensor, float32: A 4D tensor that holds the activations after
                             bilinear upsampling of the input.
        """

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 2, 3, 1))

        new_size = (target_shape[self._dims_axis[0]] * factor,
                    target_shape[self._dims_axis[1]] * factor)
        stack = tf.image.resize(stack, new_size, method="bilinear")

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 3, 1, 2))

        return stack

    def _normalize(self, maps, eps=1e-7):
        """This function normalizes the output values to a range
           between 0 and 1 per saliency map.

        Args:
            maps (tensor, float32): A 4D tensor that holds the model output.
            eps (scalar, float, optional): A small factor to avoid numerical
                                           instabilities. Defaults to 1e-7.

        Returns:
            tensor, float32: Normalized saliency maps.
        """

        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keepdims=True)
        maps = maps - min_per_image

        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keepdims=True)
        maps = maps / (eps + max_per_image)

        return maps

    def call(self, images, training=False):
        """Forward pass through the network.

        Args:
            images (tensor, float32): A 4D tensor that holds the values of the
                                      raw input images.
            training (bool): Whether in training mode.

        Returns:
            tensor, float32: A 4D tensor that holds the values of the
                             predicted saliency maps.
        """

        # Preprocess: subtract ImageNet mean
        imagenet_mean = tf.constant([103.939, 116.779, 123.68])
        imagenet_mean = tf.reshape(imagenet_mean, [1, 1, 1, 3])
        x = images - imagenet_mean

        if self._data_format == "channels_first":
            x = tf.transpose(x, (0, 3, 1, 2))

        # Encoder
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        layer10 = self.pool3(x)

        x = self.conv4_1(layer10)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        layer14 = self.pool4(x)

        x = self.conv5_1(layer14)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        layer18 = self.pool5(x)

        encoder_output = tf.concat([layer10, layer14, layer18],
                                   axis=self._channel_axis)

        # ASPP
        branch1 = self.aspp_conv1(encoder_output)
        branch2 = self.aspp_conv2(encoder_output)
        branch3 = self.aspp_conv3(encoder_output)
        branch4 = self.aspp_conv4(encoder_output)

        branch5 = tf.reduce_mean(encoder_output,
                                 axis=self._dims_axis,
                                 keepdims=True)
        branch5 = self.aspp_conv5(branch5)

        features_shape = tf.shape(encoder_output)
        branch5 = self._upsample(branch5, features_shape, 1)

        context = tf.concat([branch1, branch2, branch3, branch4, branch5],
                            axis=self._channel_axis)

        aspp_output = self.aspp_conv_out(context)

        # Decoder
        shape = tf.shape(aspp_output)
        x = self._upsample(aspp_output, shape, 2)
        x = self.decoder_conv1(x)

        shape = tf.shape(x)
        x = self._upsample(x, shape, 2)
        x = self.decoder_conv2(x)

        shape = tf.shape(x)
        x = self._upsample(x, shape, 2)
        x = self.decoder_conv3(x)

        decoder_output = self.decoder_conv4(x)

        if self._data_format == "channels_first":
            decoder_output = tf.transpose(decoder_output, (0, 2, 3, 1))

        # Normalize output
        output = self._normalize(decoder_output)

        return output

    def save_weights(self, dataset, path, device):
        """Save model weights to disk or GCS.

        Args:
            dataset (str): The dataset name.
            path (str): The path used for saving the model (local or gs://).
            device (str): Represents either "cpu", "gpu", or "tpu".
        """

        tf.io.gfile.makedirs(path)
        weights_path = path + "model_%s_%s.weights.h5" % (dataset, device)
        super().save_weights(weights_path)

    def restore(self, dataset, paths, device):
        """This function allows continued training from a prior checkpoint and
           training from scratch with the pretrained VGG16 weights. In case the
           desired dataset is not SALICON, a prior checkpoint based on the
           SALICON dataset is required.

        Args:
            dataset (str): The dataset used for training.
            paths (dict, str): A dictionary with all path elements.
            device (str): Represents either "cpu", "gpu", or "tpu".
        """

        model_name = "model_%s_%s" % (dataset, device)
        salicon_name = "model_salicon_%s" % device

        weights_ext = ".weights.h5"

        if tf.io.gfile.exists(paths["latest"] + model_name + weights_ext):
            self.load_weights(paths["latest"] + model_name + weights_ext)
            print(">> Restored weights from latest checkpoint")
        elif dataset in ("mit1003", "cat2000", "dutomron",
                         "pascals", "osie", "fiwi", "fixationadd1000"):
            if tf.io.gfile.exists(paths["best"] + salicon_name + weights_ext):
                self.load_weights(paths["best"] + salicon_name + weights_ext)
                print(">> Restored weights from SALICON checkpoint")
            else:
                raise FileNotFoundError("Train model on SALICON first")
        else:
            # Try to load VGG16 pretrained weights
            vgg16_weights_path = paths["weights"] + "vgg16_weights.h5"
            if not tf.io.gfile.exists(vgg16_weights_path):
                download.download_pretrained_weights(paths["weights"], "vgg16_weights")

            if tf.io.gfile.exists(vgg16_weights_path):
                self._load_vgg16_weights(vgg16_weights_path)
                print(">> Loaded VGG16 pretrained weights")
            else:
                print(">> Starting with random weights")

    def _load_vgg16_weights(self, weights_path):
        """Load pretrained VGG16 weights for the encoder layers.

        Args:
            weights_path (str): Path to the VGG16 weights file.
        """

        # Map our layer names to VGG16 layer names
        vgg16_layer_mapping = {
            "conv1_conv1_1": "block1_conv1",
            "conv1_conv1_2": "block1_conv2",
            "conv2_conv2_1": "block2_conv1",
            "conv2_conv2_2": "block2_conv2",
            "conv3_conv3_1": "block3_conv1",
            "conv3_conv3_2": "block3_conv2",
            "conv3_conv3_3": "block3_conv3",
            "conv4_conv4_1": "block4_conv1",
            "conv4_conv4_2": "block4_conv2",
            "conv4_conv4_3": "block4_conv3",
            "conv5_conv5_1": "block5_conv1",
            "conv5_conv5_2": "block5_conv2",
            "conv5_conv5_3": "block5_conv3",
        }

        # Load VGG16 model to get weights
        vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

        for our_name, vgg_name in vgg16_layer_mapping.items():
            our_layer = None
            for layer in self.layers:
                if layer.name == our_name:
                    our_layer = layer
                    break

            if our_layer is not None:
                vgg_layer = vgg16.get_layer(vgg_name)
                our_layer.set_weights(vgg_layer.get_weights())

    def export_saved_model(self, dataset, path, device):
        """Export the model as a SavedModel for inference.

        Args:
            dataset (str): The dataset name.
            path (str): The path used for saving the model (local or gs://).
            device (str): Represents either "cpu", "gpu", or "tpu".
        """

        model_name = "model_%s_%s" % (dataset, device)
        model_path = path + model_name

        tf.io.gfile.makedirs(model_path)

        # Create a concrete function for serving
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],
                                                     dtype=tf.float32, name="input")])
        def serving_fn(input_tensor):
            output = self(input_tensor, training=False)
            return {"output": output}

        # Save the model
        tf.saved_model.save(
            self,
            model_path,
            signatures={"serving_default": serving_fn}
        )

        print(">> Exported SavedModel to %s" % model_path)
