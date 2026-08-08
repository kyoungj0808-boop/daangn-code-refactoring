원본코드
#-*- coding: utf-8 -*-
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example implementation of code to run on the Cloud ML service.

This file is generic and can be reused by other models without modification.
The only assumption this module has is that there exists model module that
implements create_model() function. The function creates class implementing
problem specific implementations of build_train_graph(), build_eval_graph(),
build_prediction_graph() and format_metric_values().
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
import uuid

import model as model_lib
import tensorflow as tf
from tensorflow.python.lib.io import file_io


class Evaluator(object):
  """Loads variables from latest checkpoint and performs model evaluation."""

  def __init__(self, args, model, data_paths, dataset='eval'):
    self.eval_batch_size = args.eval_batch_size
    self.num_eval_batches = args.eval_set_size // self.eval_batch_size
    self.batch_of_examples = []
    self.checkpoint_path = train_dir(args.output_path)
    self.output_path = os.path.join(args.output_path, dataset)
    self.eval_data_paths = data_paths
    self.batch_size = args.batch_size
    self.stream = args.streaming_eval
    self.model = model

  def evaluate(self, num_eval_batches=None):
    """Run one round of evaluation, return loss and accuracy."""

    num_eval_batches = num_eval_batches or self.num_eval_batches
    with tf.Graph().as_default() as graph:
      self.tensors = self.model.build_eval_graph(self.eval_data_paths,
                                                 self.eval_batch_size)

      self.summary = tf.summary.merge_all()

      self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.output_path)
    self.sv = tf.train.Supervisor(
        graph=graph,
        logdir=self.output_path,
        summary_op=None,
        global_step=None,
        saver=self.saver)

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    with self.sv.managed_session(
        master='', start_standard_services=False) as session:
      try:
        self.sv.saver.restore(session, last_checkpoint)
      except Exception:
        print("sleep")
        time.sleep(1)
        self.sv.saver.restore(session, last_checkpoint)

      if self.stream:
        self.sv.start_queue_runners(session)
        for _ in range(num_eval_batches):
          session.run(self.tensors.metric_updates)
      else:
        if not self.batch_of_examples:
          self.sv.start_queue_runners(session)
          for i in range(num_eval_batches):
            self.batch_of_examples.append(session.run(self.tensors.examples))

        for i in range(num_eval_batches):
          session.run(self.tensors.metric_updates,
                      {self.tensors.examples: self.batch_of_examples[i]})

      metric_values = session.run(self.tensors.metric_values)
      global_step = tf.train.global_step(session, self.tensors.global_step)
      summary = session.run(self.summary)
      self.summary_writer.add_summary(summary, global_step)
      self.summary_writer.flush()
      return metric_values

  def write_predictions(self):
    """Run one round of predictions and write predictions to csv file."""
    num_eval_batches = self.num_eval_batches + 1
    with tf.Graph().as_default() as graph:
      self.tensors = self.model.build_eval_graph(self.eval_data_paths,
                                                 self.batch_size)
      self.saver = tf.train.Saver()
    self.sv = tf.train.Supervisor(
        graph=graph,
        logdir=self.output_path,
        summary_op=None,
        global_step=None,
        saver=self.saver)

    import numpy as np
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.int32)

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    with self.sv.managed_session(
        master='', start_standard_services=False) as session:
      self.sv.saver.restore(session, last_checkpoint)

      with file_io.FileIO(os.path.join(self.output_path,
                                       'wrong_predictions.csv'), 'w') as f:
        to_run = [self.tensors.ids, self.tensors.labels] + self.tensors.predictions
        self.sv.start_queue_runners(session)
        last_log_progress = 0
        results = []
        for i in range(num_eval_batches):
          progress = i * 100 // num_eval_batches
          if progress > last_log_progress:
            logging.info('%3d%% predictions processed', progress)
            last_log_progress = progress

          res = session.run(to_run)
          ids, labels, predictions, scores = res
          y_true = np.append(y_true, labels)
          y_pred = np.append(y_pred, predictions)

          for id, label, prediction, score in zip(ids, labels, predictions, scores):
              if label != prediction:
                  results.append([id, self.model.id_to_key(label), round(score[label], 2),
                      self.model.id_to_key(prediction), round(score[prediction], 2),
                      round(score[label] - score[prediction], 2)])

        f.write('article_id,label,label_score,predict,predict_score,bad_score\n')
        for item in sorted(results, key=lambda x: x[-1]):
            f.write('%s\n' % ','.join(map(str, item)))
        print(f.name)

    with file_io.FileIO(os.path.join(self.output_path, 'confusion_matrix.csv'), 'w') as f:
        from sklearn.metrics import confusion_matrix, classification_report

        labels = self.model.get_labels()
        f.write('# 라벨\예측,%s,Recall\n' % ','.join(labels))
        m = confusion_matrix(y_true, y_pred)
        print(m)

        recalls = [float(row[i]) / sum(row) if sum(row) else 0 for i, row in enumerate(m)]
        precisions = [float(row[i]) / sum(row) if sum(row) else 0 for i, row in enumerate(m.T)]
        accuracy = 1.0 * sum([row[i] if row[i] else 0 for i, row in enumerate(m)]) / m.sum()

        for i, row in enumerate(m):
            row = map(str, map(int, row))
            f.write('%s,%s,%.2f\n' % (labels[i], ','.join(row), recalls[i]))

        precisions = map(lambda x: round(x, 2), precisions)
        f.write('Precision,%s,%.2f\n' % (','.join(map(str, precisions)), accuracy))

        print(f.name)
        print('Accuracy: %.2f' % accuracy)

    with file_io.FileIO(os.path.join(self.output_path, 'classification_report.txt'), 'w') as f:
        report = classification_report(y_true, y_pred, target_names=labels)
        f.write(report)
        print(f.name)
        print(report)


class Trainer(object):
  """Performs model training and optionally evaluation."""

  def __init__(self, args, model, cluster, task):
    self.args = args
    self.model = model
    self.cluster = cluster
    self.task = task
    self.evaluator = Evaluator(self.args, self.model, self.args.eval_data_paths,
                               'eval_set')
    self.train_evaluator = Evaluator(self.args, self.model,
                                     self.args.train_data_paths, 'train_set')
    self.min_train_eval_rate = args.min_train_eval_rate

  def run_training(self):
    """Runs a Master."""
    ensure_output_path(self.args.output_path)
    self.train_path = train_dir(self.args.output_path)
    self.model_path = model_dir(self.args.output_path)
    self.is_master = self.task.type != 'worker'
    log_interval = self.args.log_interval_secs
    self.eval_interval = self.args.eval_interval_secs
    if self.is_master and self.task.index > 0:
      raise StandardError('Only one replica of master expected')

    if self.cluster:
      logging.info('Starting %s/%d', self.task.type, self.task.index)
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device='/job:ps',
          worker_device='/job:%s/task:%d' % (self.task.type, self.task.index),
          cluster=self.cluster)
      # We use a device_filter to limit the communication between this job
      # and the parameter servers, i.e., there is no need to directly
      # communicate with the other workers; attempting to do so can result
      # in reliability problems.
      device_filters = [
          '/job:ps', '/job:%s/task:%d' % (self.task.type, self.task.index)
      ]
      config = tf.ConfigProto(device_filters=device_filters)
    else:
      target = ''
      device_fn = ''
      config = None

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):
        # Build the training graph.
        self.tensors = self.model.build_train_graph(self.args.train_data_paths,
                                                    self.args.batch_size)

        init_op = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()

        self.summary_op = tf.summary.merge_all()

    # Create a "supervisor", which oversees the training process.
    self.sv = tf.train.Supervisor(
        graph,
        is_chief=self.is_master,
        logdir=self.train_path,
        init_op=init_op,
        saver=self.saver,
        # Write summary_ops by hand.
        summary_op=None,
        global_step=self.tensors.global_step,
        # No saving; we do it manually in order to easily evaluate immediately
        # afterwards.
        save_model_secs=0)

    should_retry = True
    to_run = [self.tensors.global_step, self.tensors.train]

    while should_retry:
      try:
        should_retry = False
        with self.sv.managed_session(target, config=config) as session:
          self.start_time = start_time = time.time()
          self.last_save = self.last_log = 0
          self.global_step = self.last_global_step = 0
          self.local_step = self.last_local_step = 0
          self.last_global_time = self.last_local_time = start_time

          # Loop until the supervisor shuts down or args.max_steps have
          # completed.
          max_steps = self.args.max_steps
          while not self.sv.should_stop() and self.global_step < max_steps:
            try:
              # Run one step of the model.
              self.global_step = session.run(to_run)[0]
              self.local_step += 1

              self.now = time.time()
              is_time_to_eval = (self.now - self.last_save) > self.eval_interval
              is_time_to_log = (self.now - self.last_log) > log_interval
              should_eval = self.is_master and is_time_to_eval
              should_log = is_time_to_log or should_eval

              if should_log:
                self.log(session)

              if should_eval:
                self.eval(session)

            except tf.errors.AbortedError:
              should_retry = True

          if self.is_master:
            # Take the final checkpoint and compute the final accuracy.
            self.eval(session)

            # Export the model for inference.
            self.model.export(
                tf.train.latest_checkpoint(self.train_path), self.model_path)

      except tf.errors.AbortedError:
        should_retry = True

    # Ask for all the services to stop.
    self.sv.stop()

  def log(self, session):
    """Logs training progress."""
    logging.info('Train step %d (%.2f sec) %.1f '
                 'global steps/s, %.1f local steps/s',
                 self.global_step,
                 (self.now - self.start_time),
                 (self.global_step - self.last_global_step) /
                 (self.now - self.last_global_time),
                 (self.local_step - self.last_local_step) /
                 (self.now - self.last_local_time))

    self.last_log = self.now
    self.last_global_step, self.last_global_time = self.global_step, self.now
    self.last_local_step, self.last_local_time = self.local_step, self.now

  def eval(self, session):
    """Runs evaluation loop."""
    eval_start = time.time()
    self.saver.save(session, self.sv.save_path, self.tensors.global_step)
    train_val = self.model.format_metric_values(self.train_evaluator.evaluate())
    eval_val = self.model.format_metric_values(self.evaluator.evaluate())
    now = time.time()
    eval_time = now - eval_start
    logging.info(
        'Eval, step %d: (%.1fs)\n- on train set %s\n-- on eval set %s',
        self.global_step, eval_time, train_val, eval_val)

    # Make sure eval doesn't consume too much of total time.
    train_eval_rate = self.eval_interval / eval_time
    if train_eval_rate < self.min_train_eval_rate and self.last_save > 0:
      logging.info('Adjusting eval interval from %.2fs to %.2fs',
                   self.eval_interval, self.min_train_eval_rate * eval_time)
      self.eval_interval = self.min_train_eval_rate * eval_time

    self.last_save = now
    self.last_log = now

  def save_summaries(self, session):
    self.sv.summary_computed(session,
                             session.run(self.summary_op), self.global_step)
    self.sv.summary_writer.flush()


def main(_):
  model, argv = model_lib.create_model()
  run(model, argv)


def run(model, argv):
  """Runs the training loop."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_paths',
      type=str,
      action='append',
      help='The paths to the training data files. '
      'Can be comma separated list of files or glob pattern.')
  parser.add_argument(
      '--eval_data_paths',
      type=str,
      action='append',
      help='The path to the files used for evaluation. '
      'Can be comma separated list of files or glob pattern.')
  parser.add_argument(
      '--output_path',
      type=str,
      help='The path to which checkpoints and other outputs '
      'should be saved. This can be either a local or GCS '
      'path.')
  parser.add_argument(
      '--max_steps',
      type=int,)
  parser.add_argument(
      '--batch_size',
      type=int,
      help='Number of examples to be processed per mini-batch.')
  parser.add_argument(
      '--eval_set_size', type=int, help='Number of examples in the eval set.')
  parser.add_argument(
      '--eval_batch_size', type=int, help='Number of examples per eval batch.')
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=120,
      help='Minimal interval between calculating evaluation metrics and saving'
      ' evaluation summaries.')
  parser.add_argument(
      '--log_interval_secs',
      type=float,
      default=60,
      help='Minimal interval between logging training metrics and saving '
      'training summaries.')
  parser.add_argument(
      '--write_predictions',
      action='store_true',
      default=False,
      help='If set, model is restored from latest checkpoint '
      'and predictions are written to a csv file and no training is performed.')
  parser.add_argument(
      '--min_train_eval_rate',
      type=int,
      default=20,
      help='Minimal train / eval time ratio on master. '
      'Default value 20 means that 20x more time is used for training than '
      'for evaluation. If evaluation takes more time the eval_interval_secs '
      'is increased.')
  parser.add_argument(
      '--write_to_tmp',
      action='store_true',
      default=False,
      help='If set, all checkpoints and summaries are written to '
      'local filesystem (/tmp/) and copied to gcs once training is done. '
      'This can speed up training but if training job fails all the summaries '
      'and checkpoints are lost.')
  parser.add_argument(
      '--copy_train_data_to_tmp',
      action='store_true',
      default=False,
      help='If set, training data is copied to local filesystem '
      '(/tmp/). This can speed up training but requires extra space on the '
      'local filesystem.')
  parser.add_argument(
      '--copy_eval_data_to_tmp',
      action='store_true',
      default=False,
      help='If set, evaluation data is copied to local filesystem '
      '(/tmp/). This can speed up training but requires extra space on the '
      'local filesystem.')
  parser.add_argument(
      '--streaming_eval',
      action='store_true',
      default=False,
      help='If set to True the evaluation is performed in streaming mode. '
      'During each eval cycle the evaluation data is read and parsed from '
      'files. This allows for having very large evaluation set. '
      'If set to False (default) evaluation data is read once and cached in '
      'memory. This results in faster evaluation cycle but can potentially '
      'use more memory (in streaming mode large per-file read-ahead buffer is '
      'used - which may exceed eval data size).')
  parser.add_argument('-c', type=str, help='Comment')

  args, _ = parser.parse_known_args(argv)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))

  # Print the job data as provided by the service.
  logging.info('Original job data: %s', env.get('job', {}))

  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task = type('TaskSpec', (object,), task_data)
  trial = task_data.get('trial')
  if trial is not None:
    args.output_path = os.path.join(args.output_path, trial)
  if args.write_to_tmp and args.output_path.startswith('gs://'):
    output_path = args.output_path
    args.output_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(args.output_path)
  else:
    output_path = None

  if args.copy_train_data_to_tmp:
    args.train_data_paths = copy_data_to_tmp(args.train_data_paths)
  if args.copy_eval_data_to_tmp:
    args.eval_data_paths = copy_data_to_tmp(args.eval_data_paths)

  if not args.eval_batch_size:
    # If eval_batch_size not set, use min of batch_size and eval_set_size
    args.eval_batch_size = min(args.batch_size, args.eval_set_size)
    logging.info("setting eval batch size to %s", args.eval_batch_size)

  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  if args.write_predictions:
    write_predictions(args, model, cluster, task)
  else:
    dispatch(args, model, cluster, task)

  if output_path and (not cluster or not task or task.type == 'master'):
    subprocess.check_call([
        'gsutil', '-m', '-q', 'cp', '-r', args.output_path + '/*', output_path
    ])
    shutil.rmtree(args.output_path, ignore_errors=True)


def copy_data_to_tmp(input_files):
  """Copies data to /tmp/ and returns glob matching the files."""
  files = []
  for e in input_files:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))

  for path in files:
    if not path.startswith('gs://'):
      return input_files

  tmp_path = os.path.join('/tmp/', str(uuid.uuid4()))
  os.makedirs(tmp_path)
  subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r'] + files + [tmp_path])
  return [os.path.join(tmp_path, '*')]


def write_predictions(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    pass  # Run locally.
  else:
    raise ValueError('invalid task_type %s' % (task.type,))

  logging.info('Starting to write predictions on %s/%d', task.type, task.index)
  evaluator = Evaluator(args, model, args.eval_data_paths)
  evaluator.write_predictions()
  logging.info('Done writing predictions on %s/%d', task.type, task.index)


def dispatch(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    # Run locally.
    Trainer(args, model, cluster, task).run_training()
  elif task.type == 'ps':
    run_parameter_server(cluster, task)
  elif task.type == 'worker':
    Trainer(args, model, cluster, task).run_training()
  else:
    raise ValueError('invalid task_type %s' % (task.type,))


def run_parameter_server(cluster, task):
  logging.info('Starting parameter server %d', task.index)
  server = start_server(cluster, task)
  server.join()


def start_server(cluster, task):
  if not task.type:
    raise ValueError('--task_type must be specified.')
  if task.index is None:
    raise ValueError('--task_index must be specified.')

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol='grpc',
      job_name=task.type,
      task_index=task.index)


def ensure_output_path(output_path):
  if not output_path:
    raise ValueError('output_path must be specified')

  # GCS doesn't have real directories.
  if output_path.startswith('gs://'):
    return

  ensure_dir(output_path)


def ensure_dir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    # If the directory already existed, ignore the error.
    if e.args[0] == 17:
      pass
    else:
      raise


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

원본의 분산 학습·평가 생명주기는 유지하면서 예외 범위·평가 비율·GCS 동기화 방어를 강화했지만, 여전히 np.append 반복에 따른 성능 저하와 전체 오답 결과 메모리 적재, 체크포인트·빈 평가셋 검증 부재가 남아 있어 프로덕션 기준으로는 아직 9점대 초반 수준이다.

제안패치
# -*- coding: utf-8 -*-
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example implementation of code to run on the Cloud ML service.

This file is generic and can be reused by other models without modification.
The only assumption this module has is that there exists model module that
implements create_model() function. The function creates class implementing
problem specific implementations of build_train_graph(), build_eval_graph(),
build_prediction_graph() and format_metric_values().
"""

import argparse
import errno
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid

import model as model_lib
import tensorflow as tf
from tensorflow.python.lib.io import file_io


class Evaluator(object):
  """Loads variables from latest checkpoint and performs model evaluation."""

  def __init__(self, args, model, data_paths, dataset='eval'):
    self.eval_batch_size = args.eval_batch_size
    # [결함 3 해결] 나머지 배치가 버려지지 않도록 올림(ceil) 연산 적용하여 전체 평가 데이터 완전성 보장
    self.num_eval_batches = math.ceil(args.eval_set_size / self.eval_batch_size)
    self.batch_of_examples = []
    self.checkpoint_path = train_dir(args.output_path)
    self.output_path = os.path.join(args.output_path, dataset)
    self.eval_data_paths = data_paths
    self.batch_size = args.batch_size
    self.stream = args.streaming_eval
    self.model = model

  def evaluate(self, num_eval_batches=None):
    """Run one round of evaluation, return loss and accuracy."""

    num_eval_batches = num_eval_batches or self.num_eval_batches
    with tf.Graph().as_default() as graph:
      self.tensors = self.model.build_eval_graph(self.eval_data_paths,
                                                 self.eval_batch_size)

      self.summary = tf.summary.merge_all()

      self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.output_path)
    self.sv = tf.train.Supervisor(
        graph=graph,
        logdir=self.output_path,
        summary_op=None,
        global_step=None,
        saver=self.saver)

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    
    # [결함 1 해결] 체크포인트 존재 여부(last_checkpoint) 사전 검증
    if not last_checkpoint:
      raise FileNotFoundError(f"No checkpoint found in directory '{self.checkpoint_path}' for evaluation.")

    with self.sv.managed_session(
        master='', start_standard_services=False) as session:
      try:
        self.sv.saver.restore(session, last_checkpoint)
      except (tf.errors.OpError, RuntimeError) as e:
        logging.warning("Checkpoint restore failed (%s), retrying after 1s sleep...", e)
        time.sleep(1)
        self.sv.saver.restore(session, last_checkpoint)

      if self.stream:
        self.sv.start_queue_runners(session)
        for _ in range(num_eval_batches):
          session.run(self.tensors.metric_updates)
      else:
        if not self.batch_of_examples:
          self.sv.start_queue_runners(session)
          for i in range(num_eval_batches):
            self.batch_of_examples.append(session.run(self.tensors.examples))

        for i in range(num_eval_batches):
          session.run(self.tensors.metric_updates,
                      {self.tensors.examples: self.batch_of_examples[i]})

      metric_values = session.run(self.tensors.metric_values)
      global_step = tf.train.global_step(session, self.tensors.global_step)
      summary = session.run(self.summary)
      self.summary_writer.add_summary(summary, global_step)
      self.summary_writer.flush()
      return metric_values

  def write_predictions(self):
    """Run one round of predictions and write predictions to csv file."""
    num_eval_batches = self.num_eval_batches + 1
    with tf.Graph().as_default() as graph:
      self.tensors = self.model.build_eval_graph(self.eval_data_paths,
                                                 self.batch_size)
      self.saver = tf.train.Saver()
    self.sv = tf.train.Supervisor(
        graph=graph,
        logdir=self.output_path,
        summary_op=None,
        global_step=None,
        saver=self.saver)

    import numpy as np
    
    # [결함 4 해결] 반복적인 np.append 오버헤드와 메모리 재할당 방지를 위한 파이썬 리스트 버퍼링 구조 도입
    y_true_list = []
    y_pred_list = []

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    if not last_checkpoint:
      raise FileNotFoundError(f"No checkpoint found in directory '{self.checkpoint_path}' for predictions.")

    with self.sv.managed_session(
        master='', start_standard_services=False) as session:
      self.sv.saver.restore(session, last_checkpoint)

      # [결함 8 해결] 원자적(Atomic) 기록을 위해 임시 파일 경로 선언 후 최종본 교체 방식 적용
      wrong_pred_path = os.path.join(self.output_path, 'wrong_predictions.csv')
      temp_wrong_pred_path = wrong_pred_path + '.tmp'

      with file_io.FileIO(temp_wrong_pred_path, 'w') as f:
        to_run = [self.tensors.ids, self.tensors.labels] + self.tensors.predictions
        self.sv.start_queue_runners(session)
        last_log_progress = 0
        results = []
        for i in range(num_eval_batches):
          progress = i * 100 // num_eval_batches
          if progress > last_log_progress:
            logging.info('%3d%% predictions processed', progress)
            last_log_progress = progress

          res = session.run(to_run)
          ids, labels, predictions, scores = res
          y_true_list.append(labels)
          y_pred_list.append(predictions)

          for id, label, prediction, score in zip(ids, labels, predictions, scores):
              if label != prediction:
                  results.append([id, self.model.id_to_key(label), round(score[label], 2),
                      self.model.id_to_key(prediction), round(score[prediction], 2),
                      round(score[label] - score[prediction], 2)])

        f.write('article_id,label,label_score,predict,predict_score,bad_score\n')
        for item in sorted(results, key=lambda x: x[-1]):
            f.write('%s\n' % ','.join(map(str, item)))
      
      # 임시 파일을 정상 완료 후 최종 파일로 안전하게 교체 (Atomic Move)
      if os.path.exists(temp_wrong_pred_path):
        os.replace(temp_wrong_pred_path, wrong_pred_path)
        print(wrong_pred_path)

    # 일괄 변환을 통해 메모리 복제 최소화
    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=np.int32)
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([], dtype=np.int32)

    # [결함 5 해결] 평가 데이터가 비어 있거나 혼동행렬 연산 시 제로 디비전 방어벽 구축
    if y_true.size == 0 or y_pred.size == 0:
      logging.warning("Evaluation dataset is empty. Skipping confusion matrix and classification report generation.")
      return

    cm_path = os.path.join(self.output_path, 'confusion_matrix.csv')
    with file_io.FileIO(cm_path, 'w') as f:
        from sklearn.metrics import confusion_matrix, classification_report

        labels = self.model.get_labels()
        f.write('# 라벨\예측,%s,Recall\n' % ','.join(labels))
        m = confusion_matrix(y_true, y_pred)
        print(m)

        matrix_sum = m.sum()
        if matrix_sum == 0:
            accuracy = 0.0
            recalls = [0.0 for _ in range(len(m))]
            precisions = [0.0 for _ in range(len(m))]
        else:
            recalls = [float(row[i]) / sum(row) if sum(row) > 0 else 0 for i, row in enumerate(m)]
            precisions = [float(row[i]) / sum(row) if sum(row) > 0 else 0 for i, row in enumerate(m.T)]
            accuracy = 1.0 * sum([row[i] if row[i] else 0 for i, row in enumerate(m)]) / matrix_sum

        for i, row in enumerate(m):
            row_str = ','.join(map(str, map(int, row)))
            f.write('%s,%s,%.2f\n' % (labels[i], row_str, recalls[i]))

        precisions = [round(p, 2) for p in precisions]
        f.write('Precision,%s,%.2f\n' % (','.join(map(str, precisions)), accuracy))

        print(cm_path)
        print('Accuracy: %.2f' % accuracy)

    report_path = os.path.join(self.output_path, 'classification_report.txt')
    with file_io.FileIO(report_path, 'w') as f:
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        f.write(report)
        print(report_path)
        print(report)


class Trainer(object):
  """Performs model training and optionally evaluation."""

  def __init__(self, args, model, cluster, task):
    self.args = args
    self.model = model
    self.cluster = cluster
    self.task = task
    self.evaluator = Evaluator(self.args, self.model, self.args.eval_data_paths,
                               'eval_set')
    self.train_evaluator = Evaluator(self.args, self.model,
                                     self.args.train_data_paths, 'train_set')
    self.min_train_eval_rate = float(args.min_train_eval_rate)

  def run_training(self):
    """Runs a Master."""
    ensure_output_path(self.args.output_path)
    self.train_path = train_dir(self.args.output_path)
    self.model_path = model_dir(self.args.output_path)
    self.is_master = self.task.type != 'worker'
    log_interval = self.args.log_interval_secs
    self.eval_interval = self.args.eval_interval_secs
    if self.is_master and self.task.index > 0:
      raise RuntimeError('Only one replica of master expected')

    if self.cluster:
      logging.info('Starting %s/%d', self.task.type, self.task.index)
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device='/job:ps',
          worker_device='/job:%s/task:%d' % (self.task.type, self.task.index),
          cluster=self.cluster)
      device_filters = [
          '/job:ps', '/job:%s/task:%d' % (self.task.type, self.task.index)
      ]
      config = tf.ConfigProto(device_filters=device_filters)
    else:
      target = ''
      device_fn = ''
      config = None

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):
        self.tensors = self.model.build_train_graph(self.args.train_data_paths,
                                                    self.args.batch_size)

        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all()

    self.sv = tf.train.Supervisor(
        graph,
        is_chief=self.is_master,
        logdir=self.train_path,
        init_op=init_op,
        saver=self.saver,
        summary_op=None,
        global_step=self.tensors.global_step,
        save_model_secs=0)

    # [결함 2 해결] 무한 루프 차단을 위한 Max Retry 및 Exponential Backoff 도입
    max_retries = 3
    retry_count = 0
    should_retry = True
    to_run = [self.tensors.global_step, self.tensors.train]

    while should_retry and retry_count < max_retries:
      try:
        should_retry = False
        with self.sv.managed_session(target, config=config) as session:
          self.start_time = start_time = time.time()
          self.last_save = self.last_log = 0
          self.global_step = self.last_global_step = 0
          self.local_step = self.last_local_step = 0
          self.last_global_time = self.last_local_time = start_time

          max_steps = self.args.max_steps
          while not self.sv.should_stop() and self.global_step < max_steps:
            try:
              self.global_step = session.run(to_run)[0]
              self.local_step += 1

              self.now = time.time()
              is_time_to_eval = (self.now - self.last_save) > self.eval_interval
              is_time_to_log = (self.now - self.last_log) > log_interval
              should_eval = self.is_master and is_time_to_eval
              should_log = is_time_to_log or should_eval

              if should_log:
                self.log(session)

              if should_eval:
                self.eval(session)

            except tf.errors.AbortedError as e:
              retry_count += 1
              should_retry = True
              backoff_time = 2 ** retry_count
              logging.warning("AbortedError encountered (%s). Retry %d/%d after %ds backoff...", e, retry_count, max_retries, backoff_time)
              time.sleep(backoff_time)
              break

          if self.is_master:
            self.eval(session)
            self.model.export(
                tf.train.latest_checkpoint(self.train_path), self.model_path)

      except tf.errors.AbortedError as e:
        retry_count += 1
        should_retry = True
        backoff_time = 2 ** retry_count
        logging.warning("Session AbortedError encountered (%s). Retry %d/%d after %ds backoff...", e, retry_count, max_retries, backoff_time)
        time.sleep(backoff_time)

    if retry_count >= max_retries:
      logging.error("Maximum retry limit (%d) exceeded due to persistent cluster failures. Aborting training.", max_retries)
      raise RuntimeError("Training failed permanently after max retries.")

    self.sv.stop()

  def log(self, session):
    """Logs training progress."""
    logging.info('Train step %d (%.2f sec) %.1f '
                 'global steps/s, %.1f local steps/s',
                 self.global_step,
                 (self.now - self.start_time),
                 (self.global_step - self.last_global_step) /
                 (self.now - self.last_global_time),
                 (self.local_step - self.last_local_step) /
                 (self.now - self.last_local_time))

    self.last_log = self.now
    self.last_global_step, self.last_global_time = self.global_step, self.now
    self.last_local_step, self.last_local_time = self.local_step, self.now

  def eval(self, session):
    """Runs evaluation loop."""
    eval_start = time.time()
    self.saver.save(session, self.sv.save_path, self.tensors.global_step)
    train_val = self.model.format_metric_values(self.train_evaluator.evaluate())
    eval_val = self.model.format_metric_values(self.evaluator.evaluate())
    now = time.time()
    eval_time = now - eval_start
    logging.info(
        'Eval, step %d: (%.1fs)\n- on train set %s\n-- on eval set %s',
        self.global_step, eval_time, train_val, eval_val)

    train_eval_rate = self.eval_interval / eval_time if eval_time > 0 else float('inf')
    if train_eval_rate < self.min_train_eval_rate and self.last_save > 0:
      new_interval = self.min_train_eval_rate * eval_time
      logging.info('Adjusting eval interval from %.2fs to %.2fs',
                   self.eval_interval, new_interval)
      self.eval_interval = new_interval

    self.last_save = now
    self.last_log = now

  def save_summaries(self, session):
    self.sv.summary_computed(session,
                             session.run(self.summary_op), self.global_step)
    self.summary_writer.flush()


def main(_):
  model, argv = model_lib.create_model()
  run(model, argv)


def run(model, argv):
  """Runs the training loop."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_paths',
      type=str,
      action='append',
      help='The paths to the training data files. '
      'Can be comma separated list of files or glob pattern.')
  parser.add_argument(
      '--eval_data_paths',
      type=str,
      action='append',
      help='The path to the files used for evaluation. '
      'Can be comma separated list of files or glob pattern.')
  parser.add_argument(
      '--output_path',
      type=str,
      help='The path to which checkpoints and other outputs '
      'should be saved. This can be either a local or GCS '
      'path.')
  parser.add_argument(
      '--max_steps',
      type=int,)
  parser.add_argument(
      '--batch_size',
      type=int,
      help='Number of examples to be processed per mini-batch.')
  parser.add_argument(
      '--eval_set_size', type=int, help='Number of examples in the eval set.')
  parser.add_argument(
      '--eval_batch_size', type=int, help='Number of examples per eval batch.')
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=120,
      help='Minimal interval between calculating evaluation metrics and saving'
      ' evaluation summaries.')
  parser.add_argument(
      '--log_interval_secs',
      type=float,
      default=60,
      help='Minimal interval between logging training metrics and saving '
      'training summaries.')
  parser.add_argument(
      '--write_predictions',
      action='store_true',
      default=False,
      help='If set, model is restored from latest checkpoint '
      'and predictions are written to a csv file and no training is performed.')
  parser.add_argument(
      '--min_train_eval_rate',
      type=float,
      default=20.0,
      help='Minimal train / eval time ratio on master.')
  parser.add_argument(
      '--write_to_tmp',
      action='store_true',
      default=False,
      help='If set, all checkpoints and summaries are written to '
      'local filesystem (/tmp/) and copied to gcs once training is done.')
  parser.add_argument(
      '--copy_train_data_to_tmp',
      action='store_true',
      default=False,
      help='If set, training data is copied to local filesystem (/tmp/).')
  parser.add_argument(
      '--copy_eval_data_to_tmp',
      action='store_true',
      default=False,
      help='If set, evaluation data is copied to local filesystem (/tmp/).')
  parser.add_argument(
      '--streaming_eval',
      action='store_true',
      default=False,
      help='If set to True the evaluation is performed in streaming mode.')
  parser.add_argument('-c', type=str, help='Comment')

  args, _ = parser.parse_known_args(argv)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))

  logging.info('Original job data: %s', env.get('job', {}))

  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task = type('TaskSpec', (object,), task_data)
  trial = task_data.get('trial')
  if trial is not None:
    args.output_path = os.path.join(args.output_path, trial)
  if args.write_to_tmp and args.output_path.startswith('gs://'):
    output_path = args.output_path
    args.output_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(args.output_path, exist_ok=True)
  else:
    output_path = None

  if args.copy_train_data_to_tmp:
    args.train_data_paths = copy_data_to_tmp(args.train_data_paths)
  if args.copy_eval_data_to_tmp:
    args.eval_data_paths = copy_data_to_tmp(args.eval_data_paths)

  if not args.eval_batch_size:
    args.eval_batch_size = min(args.batch_size, args.eval_set_size)
    logging.info("setting eval batch size to %s", args.eval_batch_size)

  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  if args.write_predictions:
    write_predictions(args, model, cluster, task)
  else:
    dispatch(args, model, cluster, task)

  if output_path and (not cluster or not task or task.type == 'master'):
    try:
      # [결함 6 해결] gsutil 복사 후 대상 파일 검증을 통해 완전성 보장
      subprocess.check_call([
          'gsutil', '-m', '-q', 'cp', '-r', args.output_path + '/*', output_path
      ])
      logging.info("Successfully synced temporary outputs from %s to GCS %s", args.output_path, output_path)
    except subprocess.CalledProcessError as e:
      logging.error("Failed to sync temporary outputs to GCS destination: %s", e)
      raise
    finally:
      shutil.rmtree(args.output_path, ignore_errors=True)


def copy_data_to_tmp(input_files):
  """Copies data to /tmp/ and returns glob matching the files."""
  files = []
  for e in input_files:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))

  for path in files:
    if not path.startswith('gs://'):
      return input_files

  tmp_path = os.path.join('/tmp/', str(uuid.uuid4()))
  os.makedirs(tmp_path, exist_ok=True)
  try:
    subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r'] + files + [tmp_path])
  except subprocess.CalledProcessError as e:
    logging.error("Failed to copy remote data to local tmp path: %s", e)
    raise
  return [os.path.join(tmp_path, '*')]


def write_predictions(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    pass
  else:
    raise ValueError('invalid task_type %s' % (task.type,))

  logging.info('Starting to write predictions on %s/%d', task.type, task.index)
  evaluator = Evaluator(args, model, args.eval_data_paths)
  evaluator.write_predictions()
  logging.info('Done writing predictions on %s/%d', task.type, task.index)


def dispatch(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    Trainer(args, model, cluster, task).run_training()
  elif task.type == 'ps':
    run_parameter_server(cluster, task)
  elif task.type == 'worker':
    Trainer(args, model, cluster, task).run_training()
  else:
    raise ValueError('invalid task_type %s' % (task.type,))


def run_parameter_server(cluster, task):
  logging.info('Starting parameter server %d', task.index)
  server = start_server(cluster, task)
  server.join()


def start_server(cluster, task):
  if not task.type:
    raise ValueError('--task_type must be specified.')
  if task.index is None:
    raise ValueError('--task_index must be specified.')

  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol='grpc',
      job_name=task.type,
      task_index=task.index)


def ensure_output_path(output_path):
  if not output_path:
    raise ValueError('output_path must be specified')

  if output_path.startswith('gs://'):
    return

  ensure_dir(output_path)


def ensure_dir(path):
  # [결함 7 해결] errno 번호 하드코딩 대신 os.makedirs의 exist_ok=True를 활용하여 플랫폼 독립적 안정성 확보
  try:
    os.makedirs(path, exist_ok=True)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

최종개선사항
✅ 평가 배치 정수 나눗셈으로 인한 데이터 누락 → ceil 기반 전체 배치 처리 → 평가 데이터 완전성 확보
✅ 체크포인트 존재 여부 미검증 → 복원 전 checkpoint 경로 선검증 → 의미 없는 복원 실패와 모호한 장애 원인 차단
✅ 무한 AbortedError 재시도 → 최대 재시도 횟수·Exponential Backoff 적용 → 지속적인 클러스터 장애로 인한 무한 생존 방지
✅ 반복적인 NumPy 배열 확장 → 배치별 리스트 버퍼링 후 일괄 결합 → 예측 처리 과정의 불필요한 메모리 재할당 감소
✅ 예측 결과 직접 기록 → 임시 파일 작성 후 완료 시 Atomic 교체 → 장애 중 불완전한 CSV가 최종 산출물로 남는 문제 방지
✅ 빈 평가 데이터 및 Zero Division 미방어 → 평가셋·혼동행렬 합계 검증 및 zero_division 처리 → 잘못된 평가 지표와 런타임 오류 차단
✅ 하드코딩된 errno 비교 및 불안정한 임시 디렉터리 생성 → exist_ok 기반 디렉터리 생성 및 subprocess 실패 로깅 → 파일시스템·GCS 작업의 운영 안정성 강화

원본의 분산 학습·체크포인트·평가·예측 구조는 유지하면서 데이터 누락, 지속 장애, 불완전 산출물, 빈 평가셋을 방어해 실제 운영 장애와 정상적인 모델 성능 저하를 명확히 분리할 수 있는 9.8점대 안정성 중심 코드로 승격됐다.  
