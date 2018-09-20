import tensorflow as tf
import os
from models import cyclegan_model
from .helpers import convert2float

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'modelpath (.pb)')
tf.flags.DEFINE_string('input', 'input_sample.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_string('image_size', '256', 'image size,  default: 256')

def inference():
    """Translate an image to another image
    An example of command-line usage is:
    python export_graph.py --model pretrained/apple2orange.pb \
                        --input input_sample.jpg \
                        --output output_sample.jpg \
                        --image_size 256
    """
    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.GFile(FLAGS.input, 'rb') as f:
            image_data = f.read()
            input_image = tf.image.decode_jpeg(image_data, channels=3)
            input_image = tf.image.resize_images(input_image,
                size=(FLAGS.image_size, FLAGS.image_size))
            input_image = convert2float(input_image)
            input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
        
        with tf.gfile.GFile(FLAGS.model, 'rb') as mf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(mf.read())
        [output_image] = tf.import_graph_def(
            graph_def, input_map={'input_image':input_image},
            return_elements=['output_image:0'],
            name='output'
        )

    with tf.Session(graph=graph):
        generated = output_image.eval()
        with open(FLAGS.output, 'wb') as f:
            f.write(generated)


def main(unsaved_argv):
    inference()


if __name__ == '__main__':
    tf.app.run()

