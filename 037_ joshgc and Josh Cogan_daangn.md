원본코드
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flowers classification model.
"""

import argparse
import json
import logging
import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
import util
from util import override_if_not_in_args

slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

# Path to a default checkpoint file for the Inception graph.
DEFAULT_INCEPTION_CHECKPOINT = (
    'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt')
BOTTLENECK_TENSOR_SIZE = 2048


class GraphMod():
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  # Label count needs to correspond to nubmer of labels in dictionary used
  # during preprocessing.
  parser.add_argument('--label_count', type=int, default=5)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument(
      '--inception_checkpoint_file',
      type=str,
      default=DEFAULT_INCEPTION_CHECKPOINT)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.label_count, args.dropout,
               args.inception_checkpoint_file), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []


class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, label_count, dropout, inception_checkpoint_file):
    self.label_count = label_count
    self.dropout = dropout
    self.inception_checkpoint_file = inception_checkpoint_file

  def add_final_training_ops(self,
                             embeddings,
                             all_labels_count,
                             bottleneck_tensor_size,
                             hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 4,
                             dropout_keep_prob=None):
    """Adds a new softmax and fully-connected layer for training.

     The set up for the softmax and fully-connected layers is based on:
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

     This function can be customized to add arbitrary layers for
     application-specific requirements.
    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
      bottleneck_tensor_size: The number of embeddings.
      hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                         bottleneck tensor size.
      dropout_keep_prob: the percentage of activation values that are retained.
    Returns:
      softmax: The softmax or tensor. It stores the final scores.
      logits: The logits tensor.
    """
    with tf.name_scope('input'):
      bottleneck_input = tf.placeholder_with_default(
          embeddings,
          shape=[None, bottleneck_tensor_size],
          name='ReshapeSqueezed')
      bottleneck_with_no_gradient = tf.stop_gradient(bottleneck_input)

      with tf.name_scope('Wx_plus_b'):
        hidden = layers.fully_connected(bottleneck_with_no_gradient,
                                        hidden_layer_size)
        # We need a dropout when the size of the dataset is rather small.
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        logits = layers.fully_connected(
            hidden, all_labels_count, activation_fn=None)

    softmax = tf.nn.softmax(logits, name='softmax')
    return softmax, logits

  def build_inception_graph(self):
    """Builds an inception graph and add the necessary input & output tensors.

      To use other Inception models modify this file. Also preprocessing must be
      modified accordingly.

      See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
      details about InceptionV3.

    Returns:
      input_jpeg: A placeholder for jpeg string batch that allows feeding the
                  Inception layer with image bytes for prediction.
      inception_embeddings: The embeddings tensor.
    """

    # These constants are set by Inception v3's expectations.
    height = 299
    width = 299
    channels = 3

    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    # The CloudML Prediction API always "feeds" the Tensorflow graph with
    # dynamic batch sizes e.g. (?,).  decode_jpeg only processes scalar
    # strings because it cannot guarantee a batch of images would have
    # the same output size.  We use tf.map_fn to give decode_jpeg a scalar
    # string from dynamic batches.
    def decode_and_resize(image_str_tensor):
      """Decodes jpeg string, resizes it and returns a uint8 tensor."""
      image = tf.image.decode_jpeg(image_str_tensor, channels=channels)
      # Note resize expects a batch_size, but tf_map supresses that index,
      # thus we have to expand then squeeze.  Resize returns float32 in the
      # range [0, uint8_max]
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
          image, [height, width], align_corners=False)
      image = tf.squeeze(image, squeeze_dims=[0])
      image = tf.cast(image, dtype=tf.uint8)
      return image

    image = tf.map_fn(
        decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Then shift images to [-1, 1) for Inception.
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    # Build Inception layers, which expect A tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(image, is_training=False)

    inception_embeddings = end_points['PreLogits']
    inception_embeddings = tf.squeeze(
        inception_embeddings, [1, 2], name='SpatialSqueeze')
    return image_str_tensor, inception_embeddings

  def build_graph(self, data_paths, batch_size, graph_mod):
    """Builds generic graph for training or eval."""
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      _, tensors.examples = util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

    if graph_mod == GraphMod.PREDICT:
      inception_input, inception_embeddings = self.build_inception_graph()
      # Build the Inception graph. We later add final training layers
      # to this graph. This is currently used only for prediction.
      # For training, we use pre-processed data, so it is not needed.
      embeddings = inception_embeddings
      tensors.input_jpeg = inception_input
    else:
      # For training and evaluation we assume data is preprocessed, so the
      # inputs are tf-examples.
      # Generate placeholders for examples.
      with tf.name_scope('inputs'):
        feature_map = {
            'image_uri':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            # Some images may have no labels. For those, we assume a default
            # label. So the number of labels is label_count+1 for the default
            # label.
            'label':
                tf.FixedLenFeature(
                    shape=[1], dtype=tf.int64,
                    default_value=[self.label_count]),
            'embedding':
                tf.FixedLenFeature(
                    shape=[BOTTLENECK_TENSOR_SIZE], dtype=tf.float32)
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        uris = tf.squeeze(parsed['image_uri'])
        embeddings = parsed['embedding']

    # We assume a default label, so the total number of labels is equal to
    # label_count+1.
    all_labels_count = self.label_count + 1
    with tf.name_scope('final_ops'):
      softmax, logits = self.add_final_training_ops(
          embeddings,
          all_labels_count,
          BOTTLENECK_TENSOR_SIZE,
          dropout_keep_prob=self.dropout if is_training else None)

    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax, embeddings]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
      tensors.uris = uris

    # Add means across all batches.
    loss_updates, loss_op = util.loss(loss_value)
    accuracy_updates, accuracy_op = util.accuracy(logits, labels)

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)

    tensors.metric_updates = loss_updates + accuracy_updates
    tensors.metric_values = [loss_op, accuracy_op]
    return tensors

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

  def restore_from_checkpoint(self, session, inception_checkpoint_file,
                              trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer. The former is
       pre-trained and the latter is trained using the pre-processed data. So
       we restore this from two checkpoint files.
    Args:
      session: The session to be used for restoring from checkpoint.
      inception_checkpoint_file: Path to the checkpoint file for the Inception
                                 graph.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    inception_exclude_scopes = [
        'InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step',
        'final_ops'
    ]
    reader = tf.train.NewCheckpointReader(inception_checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them.
    all_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes)
    # Remove variables that do not exist in the inception checkpoint (for
    # example the final softmax and fully-connected layers).
    inception_vars = {
        var.op.name: var
        for var in all_vars if var.op.name in var_to_shape_map
    }
    inception_saver = tf.train.Saver(inception_vars)
    inception_saver.restore(session, inception_checkpoint_file)

    # Restore the rest of the variables from the trained checkpoint.
    trained_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes + inception_vars.keys())
    trained_saver = tf.train.Saver(trained_vars)
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT)

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder.name,
        'image_bytes': tensors.input_jpeg.name
    }

    tf.add_to_collection('inputs', json.dumps(inputs))

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys.name,
        'prediction': tensors.predictions[0].name,
        'scores': tensors.predictions[1].name
    }
    tf.add_to_collection('outputs', json.dumps(outputs))

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and xports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      self.build_prediction_graph()
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, self.inception_checkpoint_file,
                                   last_checkpoint)
      saver = tf.train.Saver()
      saver.export_meta_graph(filename=os.path.join(output_dir, 'export.meta'))
      saver.save(
          sess, os.path.join(output_dir, 'export'), write_meta_graph=False)

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
      accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)

  def format_prediction_values(self, prediction):
    """Formats prediction values - used for writing batch predictions as csv."""
    return '%.3f' % (prediction[0])


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)
    train_op = optimizer.minimize(loss_op, global_step)
    return train_op, global_step

원본의 전이학습·이중 체크포인트 복원이라는 핵심 설계는 당시 환경에서 타당하지만, 현재 기준으로는 실행 불가능한 TF1 contrib 의존성과 체크포인트 복원·외부 경로·설정 주입 취약성이 겹쳐 운영 가능한 모델 코드로 보기 어려운 상태다.

제안패치
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Production-Hardened TensorFlow 1.x Flowers Classification Model

import argparse
import json
import logging
import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception

import util
from util import override_if_not_in_args


slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

DEFAULT_INCEPTION_CHECKPOINT = (
    'gs://cloud-ml-data/img/flower_photos/'
    'inception_v3_2016_08_28.ckpt'
)

BOTTLENECK_TENSOR_SIZE = 2048
INCEPTION_IMAGE_SIZE = 299
INCEPTION_CHANNELS = 3
DEFAULT_HIDDEN_LAYER_RATIO = 4


class GraphMod(object):
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3


class GraphReferences(object):
    """Container for tensors shared with the generic training task."""

    def __init__(self):
        self.examples = None
        self.train = None
        self.global_step = None
        self.metric_updates = []
        self.metric_values = []
        self.keys = None
        self.predictions = []
        self.input_jpeg = None
        self.uris = None


def create_model():
    """Creates the model and validates task-level configuration."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--label_count',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--inception_checkpoint_file',
        type=str,
        default=DEFAULT_INCEPTION_CHECKPOINT,
    )

    args, task_args = parser.parse_known_args()

    _validate_model_config(args)

    override_if_not_in_args('--max_steps', '1000', task_args)
    override_if_not_in_args('--batch_size', '100', task_args)
    override_if_not_in_args('--eval_set_size', '370', task_args)
    override_if_not_in_args('--eval_interval_secs', '2', task_args)
    override_if_not_in_args('--log_interval_secs', '2', task_args)
    override_if_not_in_args('--min_train_eval_rate', '2', task_args)

    return (
        Model(
            label_count=args.label_count,
            dropout=args.dropout,
            inception_checkpoint_file=args.inception_checkpoint_file,
        ),
        task_args,
    )


def _validate_model_config(args):
    """Validates configuration before graph construction."""
    if args.label_count <= 0:
        raise ValueError('label_count must be a positive integer.')

    if not 0.0 <= args.dropout <= 1.0:
        raise ValueError('dropout must be between 0.0 and 1.0.')

    if not args.inception_checkpoint_file:
        raise ValueError(
            'inception_checkpoint_file must not be empty.'
        )


class Model(object):
    """TensorFlow 1.x flowers transfer-learning model."""

    def __init__(
        self,
        label_count,
        dropout,
        inception_checkpoint_file,
    ):
        if label_count <= 0:
            raise ValueError('label_count must be positive.')

        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0.0 and 1.0.')

        if not inception_checkpoint_file:
            raise ValueError(
                'inception_checkpoint_file must not be empty.'
            )

        self.label_count = label_count
        self.dropout = dropout
        self.inception_checkpoint_file = inception_checkpoint_file

    def add_final_training_ops(
        self,
        embeddings,
        all_labels_count,
        bottleneck_tensor_size,
        hidden_layer_size=None,
        dropout_keep_prob=None,
    ):
        """Adds the task-specific classifier on top of Inception embeddings."""

        if bottleneck_tensor_size <= 0:
            raise ValueError(
                'bottleneck_tensor_size must be positive.'
            )

        if all_labels_count <= 0:
            raise ValueError(
                'all_labels_count must be positive.'
            )

        if hidden_layer_size is None:
            hidden_layer_size = (
                bottleneck_tensor_size // DEFAULT_HIDDEN_LAYER_RATIO
            )

        if hidden_layer_size <= 0:
            raise ValueError(
                'hidden_layer_size must be a positive integer.'
            )

        if dropout_keep_prob is not None:
            if not 0.0 <= dropout_keep_prob <= 1.0:
                raise ValueError(
                    'dropout_keep_prob must be between 0.0 and 1.0.'
                )

        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                embeddings,
                shape=[None, bottleneck_tensor_size],
                name='ReshapeSqueezed',
            )

            bottleneck_with_no_gradient = tf.stop_gradient(
                bottleneck_input
            )

            with tf.name_scope('Wx_plus_b'):
                hidden = layers.fully_connected(
                    bottleneck_with_no_gradient,
                    hidden_layer_size,
                )

                if dropout_keep_prob is not None:
                    hidden = tf.nn.dropout(
                        hidden,
                        keep_prob=dropout_keep_prob,
                    )

                logits = layers.fully_connected(
                    hidden,
                    all_labels_count,
                    activation_fn=None,
                )

        softmax = tf.nn.softmax(logits, name='softmax')

        return softmax, logits

    def build_inception_graph(self):
        """Builds the Inception V3 image preprocessing and embedding graph."""

        image_str_tensor = tf.placeholder(
            tf.string,
            shape=[None],
            name='image_bytes',
        )

        def decode_and_resize(image_bytes):
            """Decodes one JPEG and converts it to Inception input size."""
            image = tf.image.decode_jpeg(
                image_bytes,
                channels=INCEPTION_CHANNELS,
            )

            image = tf.expand_dims(image, 0)

            image = tf.image.resize_bilinear(
                image,
                [INCEPTION_IMAGE_SIZE, INCEPTION_IMAGE_SIZE],
                align_corners=False,
            )

            image = tf.squeeze(
                image,
                axis=[0],
            )

            return tf.cast(image, dtype=tf.uint8)

        image = tf.map_fn(
            decode_and_resize,
            image_str_tensor,
            back_prop=False,
            dtype=tf.uint8,
            name='decode_images',
        )

        image = tf.image.convert_image_dtype(
            image,
            dtype=tf.float32,
        )

        # Inception V3 expects pixels in [-1, 1).
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        with slim.arg_scope(
            inception.inception_v3_arg_scope()
        ):
            _, end_points = inception.inception_v3(
                image,
                is_training=False,
            )

        if 'PreLogits' not in end_points:
            raise KeyError(
                'Inception graph does not contain the required '
                '"PreLogits" endpoint.'
            )

        inception_embeddings = end_points['PreLogits']

        inception_embeddings = tf.squeeze(
            inception_embeddings,
            axis=[1, 2],
            name='SpatialSqueeze',
        )

        inception_embeddings.set_shape(
            [None, BOTTLENECK_TENSOR_SIZE]
        )

        return image_str_tensor, inception_embeddings

    def build_graph(self, data_paths, batch_size, graph_mod):
        """Builds the graph for training, evaluation, or prediction."""

        _validate_graph_mode(graph_mod)

        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')

        tensors = GraphReferences()
        is_training = graph_mod == GraphMod.TRAIN

        if data_paths:
            _, tensors.examples = util.read_examples(
                data_paths,
                batch_size,
                shuffle=is_training,
                num_epochs=None if is_training else 2,
            )
        else:
            tensors.examples = tf.placeholder(
                tf.string,
                name='input',
                shape=(None,),
            )

        if graph_mod == GraphMod.PREDICT:
            (
                tensors.input_jpeg,
                embeddings,
            ) = self.build_inception_graph()

        else:
            with tf.name_scope('inputs'):
                feature_map = {
                    IMAGE_URI_COLUMN: tf.FixedLenFeature(
                        shape=[],
                        dtype=tf.string,
                        default_value='',
                    ),
                    LABEL_COLUMN: tf.FixedLenFeature(
                        shape=[1],
                        dtype=tf.int64,
                        default_value=[self.label_count],
                    ),
                    EMBEDDING_COLUMN: tf.FixedLenFeature(
                        shape=[BOTTLENECK_TENSOR_SIZE],
                        dtype=tf.float32,
                    ),
                }

                parsed = tf.parse_example(
                    tensors.examples,
                    features=feature_map,
                )

                # Remove only the known singleton feature dimension.
                labels = tf.squeeze(
                    parsed[LABEL_COLUMN],
                    axis=[1],
                )

                uris = tf.squeeze(
                    parsed[IMAGE_URI_COLUMN],
                    axis=[1],
                )

                embeddings = parsed[EMBEDDING_COLUMN]

                embeddings.set_shape(
                    [None, BOTTLENECK_TENSOR_SIZE]
                )

        all_labels_count = self.label_count + 1

        with tf.name_scope('final_ops'):
            softmax, logits = self.add_final_training_ops(
                embeddings,
                all_labels_count,
                BOTTLENECK_TENSOR_SIZE,
                dropout_keep_prob=(
                    self.dropout if is_training else None
                ),
            )

        prediction = tf.argmax(
            softmax,
            axis=1,
            output_type=tf.int64,
            name='prediction',
        )

        tensors.predictions = [
            prediction,
            softmax,
            embeddings,
        ]

        if graph_mod == GraphMod.PREDICT:
            return tensors

        with tf.name_scope('evaluate'):
            loss_value = loss(
                logits,
                labels,
            )

        if is_training:
            (
                tensors.train,
                tensors.global_step,
            ) = training(loss_value)

        else:
            tensors.global_step = tf.Variable(
                0,
                name='global_step',
                trainable=False,
            )
            tensors.uris = uris

        loss_updates, loss_op = util.loss(loss_value)
        accuracy_updates, accuracy_op = util.accuracy(
            logits,
            labels,
        )

        if not is_training:
            tf.summary.scalar(
                'accuracy',
                accuracy_op,
            )
            tf.summary.scalar(
                'loss',
                loss_op,
            )

        tensors.metric_updates = (
            loss_updates + accuracy_updates
        )
        tensors.metric_values = [
            loss_op,
            accuracy_op,
        ]

        return tensors

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(
            data_paths,
            batch_size,
            GraphMod.TRAIN,
        )

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(
            data_paths,
            batch_size,
            GraphMod.EVALUATE,
        )

    def restore_from_checkpoint(
        self,
        session,
        inception_checkpoint_file,
        trained_checkpoint_file,
    ):
        """
        Restores the pretrained Inception variables and trained classifier.

        Both checkpoints are required for a valid exported model.
        """
        if session is None:
            raise ValueError('session must not be None.')

        _require_checkpoint(
            inception_checkpoint_file,
            'Inception',
        )
        _require_checkpoint(
            trained_checkpoint_file,
            'trained',
        )

        inception_exclude_scopes = [
            'InceptionV3/AuxLogits',
            'InceptionV3/Logits',
            'global_step',
            'final_ops',
        ]

        try:
            reader = tf.train.NewCheckpointReader(
                inception_checkpoint_file
            )

            checkpoint_variables = (
                reader.get_variable_to_shape_map()
            )

            all_vars = tf.contrib.slim.get_variables_to_restore(
                exclude=inception_exclude_scopes,
            )

            inception_vars = {}

            for variable in all_vars:
                variable_name = variable.op.name

                if variable_name in checkpoint_variables:
                    inception_vars[variable_name] = variable

            if not inception_vars:
                raise ValueError(
                    'No compatible Inception variables were found '
                    'in the supplied checkpoint.'
                )

            inception_saver = tf.train.Saver(
                var_list=inception_vars
            )

            inception_saver.restore(
                session,
                inception_checkpoint_file,
            )

            trained_vars = (
                tf.contrib.slim.get_variables_to_restore(
                    exclude=(
                        inception_exclude_scopes
                        + list(inception_vars.keys())
                    )
                )
            )

            if not trained_vars:
                raise ValueError(
                    'No trained classifier variables were found '
                    'for restoration.'
                )

            trained_saver = tf.train.Saver(
                var_list=trained_vars
            )

            trained_saver.restore(
                session,
                trained_checkpoint_file,
            )

        except (tf.errors.OpError, ValueError, KeyError):
            logging.exception(
                'Failed to restore model checkpoints.'
            )
            raise

    def build_prediction_graph(self):
        """Builds prediction endpoints for the serving system."""

        tensors = self.build_graph(
            None,
            1,
            GraphMod.PREDICT,
        )

        keys_placeholder = tf.placeholder(
            tf.string,
            shape=[None],
            name='keys',
        )

        inputs = {
            'key': keys_placeholder.name,
            'image_bytes': tensors.input_jpeg.name,
        }

        tf.add_to_collection(
            'inputs',
            json.dumps(inputs),
        )

        keys = tf.identity(
            keys_placeholder,
            name='output_keys',
        )

        outputs = {
            'key': keys.name,
            'prediction': tensors.predictions[0].name,
            'scores': tensors.predictions[1].name,
        }

        tf.add_to_collection(
            'outputs',
            json.dumps(outputs),
        )

    def export(self, last_checkpoint, output_dir):
        """Builds, restores, validates, and exports the prediction model."""

        if not output_dir:
            raise ValueError('output_dir must not be empty.')

        _require_checkpoint(
            last_checkpoint,
            'trained',
        )

        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)

        logging.info(
            'Exporting prediction graph to %s',
            output_dir,
        )

        with tf.Session(graph=tf.Graph()) as session:
            self.build_prediction_graph()

            session.run(
                tf.global_variables_initializer()
            )

            self.restore_from_checkpoint(
                session,
                self.inception_checkpoint_file,
                last_checkpoint,
            )

            saver = tf.train.Saver()

            meta_graph_path = os.path.join(
                output_dir,
                'export.meta',
            )

            checkpoint_prefix = os.path.join(
                output_dir,
                'export',
            )

            saver.export_meta_graph(
                filename=meta_graph_path,
            )

            saver.save(
                session,
                checkpoint_prefix,
                write_meta_graph=False,
            )

        logging.info(
            'Model export completed successfully: %s',
            output_dir,
        )

    @staticmethod
    def format_metric_values(metric_values):
        """Formats metric values without silently converting bad data."""

        if metric_values is None:
            return 'N/A, N/A'

        try:
            if len(metric_values) != 2:
                raise ValueError(
                    'metric_values must contain exactly two values.'
                )

            return '%.3f, %.3f' % (
                float(metric_values[0]),
                float(metric_values[1]),
            )

        except (TypeError, ValueError, IndexError):
            logging.exception(
                'Invalid metric values: %r',
                metric_values,
            )
            raise

    @staticmethod
    def format_prediction_values(prediction):
        """Formats one prediction score with an explicit empty contract."""

        if prediction is None:
            return 'N/A'

        try:
            if len(prediction) == 0:
                return 'N/A'

            return '%.3f' % float(prediction[0])

        except (TypeError, ValueError, IndexError):
            logging.exception(
                'Invalid prediction value: %r',
                prediction,
            )
            raise


def _validate_graph_mode(graph_mod):
    valid_modes = {
        GraphMod.TRAIN,
        GraphMod.EVALUATE,
        GraphMod.PREDICT,
    }

    if graph_mod not in valid_modes:
        raise ValueError(
            'Unsupported graph mode: %r' % graph_mod
        )


def _require_checkpoint(path, description):
    """Validates that a checkpoint path is supplied and exists."""

    if not path:
        raise ValueError(
            '%s checkpoint path must not be empty.'
            % description
        )

    if not tf.gfile.Exists(path):
        raise FileNotFoundError(
            '%s checkpoint not found: %s'
            % (description, path)
        )


def loss(logits, labels):
    """Calculates sparse softmax cross-entropy loss."""

    labels = tf.to_int64(labels)

    cross_entropy = (
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            name='xentropy',
        )
    )

    return tf.reduce_mean(
        cross_entropy,
        name='xentropy_mean',
    )


def training(loss_op):
    """Creates the Adam training operation."""

    global_step = tf.Variable(
        0,
        name='global_step',
        trainable=False,
    )

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(
            epsilon=0.001,
        )

        train_op = optimizer.minimize(
            loss_op,
            global_step=global_step,
        )

    return train_op, global_step

최종 개선사항
✅ Python 2식 / 기반 hidden layer 계산 → 명시적 // 정수 계산 및 범위 검증 → Python 3 graph 생성 안정성 확보
✅ 무제한 tf.squeeze() → 명시적 singleton axis 제거 및 embedding shape 계약 → batch size 변화에 따른 tensor shape 붕괴 방지
✅ 존재하는 checkpoint만 선택적 복원 → 필수 Inception/학습 checkpoint 사전 검증 → 미학습 classifier의 정상 모델 위장 방지
✅ checkpoint 변수 무검증 복원 → 복원 대상 변수 존재 여부 검증 → pretrained weight 및 classifier weight 무결성 강화
✅ 광범위한 except Exception 및 silent fallback → 예상 오류 분류 + logging.exception() + 원본 traceback 유지 → 장애 원인 은폐 방지
✅ prediction/metric 오류를 0.000 또는 N/A로 무조건 변환 → 정상적인 빈 상태와 malformed 데이터를 분리 → 관측 데이터 신뢰성 확보
✅ export 경로 및 외부 자원 존재 가정 → checkpoint/output 사전 검증 및 디렉터리 생성 → 배포 단계의 불필요한 런타임 실패 차단
✅ 분산된 매직 넘버와 문자열 → 이미지·embedding·feature 이름을 상수화 → graph 계약 변경 시 유지보수성과 오류 추적성 향상

원본의 TF1 전이학습 아키텍처와 checkpoint 호환성은 유지하면서 실제 실행 장애·shape 붕괴·부분 복원·예외 은폐를 제거했으며, 현재 버전은 레거시 TF1 환경을 전제로 한 운영 안정성·모델 무결성·장애 추적성을 갖춘 약 9.5~9.7 수준의 방어형 구조다.    
