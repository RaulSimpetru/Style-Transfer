import sys, getopt

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import numpy as np

unixOptions = "i:s:a:b:d:e:"
gnuOptions = ["input=", "style=", "input_weight=", "style_wight=", "max_dim=", "epochs="]


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


def load_img(path_to_img, _max_dim):
    max_dim = _max_dim
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def main(args):
    argument_list = args[1:]

    try:
        arguments, values = getopt.getopt(argument_list, unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    style_weight = 1e-1
    content_weight = 1e3
    temp_max_dim = 512
    epochs = 10
    steps = 100

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-i", "--input"):
            content_path = str(currentValue)
        elif currentArgument in ("-s", "--style"):
            style_path = str(currentValue)
        elif currentArgument in ("-a", "--input_weight"):
            content_weight = float(currentValue)
        elif currentArgument in ("-b", "--style_weight"):
            style_weight = float(currentValue)
        elif currentArgument in ("-d", "--max_dim"):
            temp_max_dim = float(currentValue)
        elif currentArgument in ("-e", "--epochs"):
            epochs = int(currentValue)

    content_image = load_img(content_path, temp_max_dim)
    style_image = load_img(style_path, temp_max_dim)

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg = vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                              outputs[self.num_style_layers:])

            style_outputs = [gram_matrix(style_output)
                             for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name: value
                          for style_name, value
                          in zip(self.style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}

    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * total_variation_loss(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    total_variation_weight = 1e8  # Choose one

    image = tf.Variable(content_image)

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps):
            step += 1
            train_step(image)
            print("Epoch: {} Step: {}".format(n + 1, m + 1))
            # print(image)

    end = time.time()
    print("Total time: {:.1f} min".format(round((end - start)/60, 2)))

    file_name = '{}_combined_with_{}.png'.format(content_path.replace(".jpg", ""), style_path.replace(".jpg", ""))
    mpl.image.imsave(file_name, np.asarray(image[0]))

    print("\nSaved as {}".format(file_name))


if __name__ == '__main__':
    main(sys.argv)
