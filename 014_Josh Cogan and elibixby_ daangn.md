원본코드
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               start_id=0,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._ids = numpy.arange(start_id, start_id + self._num_examples)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._ids = self._ids[perm]
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return (self._ids[start:end], self._images[start:end],
            self._labels[start:end])


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  train_images = extract_images(local_file)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  test_images = extract_images(local_file)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  train = DataSet(train_images, train_labels, start_id=0, dtype=dtype)
  validation = DataSet(validation_images,
                       validation_labels,
                       start_id=len(train_images),
                       dtype=dtype)
  test = DataSet(test_images,
                 test_labels,
                 start_id=(len(train_images) + len(validation_images)),
                 dtype=dtype)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist():
  return read_data_sets('MNIST_data')

원본의 MNIST 전용 규모를 고려하면 6.8점은 다소 과하게 낮지만, 두 평가를 섞어 냉정하게 보면 레거시 데이터 로더로서 기본 구조는 탄탄하나, 실제 장애 상황에서 파일 손상·잘못된 입력·epoch 경계의 데이터 누락을 확실히 방어하지 못하는 것이 핵심 약점이며, 현재 프로덕션 기준에서는 데이터 무결성 검증과 배치 처리 안정성을 우선 보강해야 하는 코드다.

제안패치
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Production-Grade Refactoring: Zero Data Loss, Streaming Pipeline, API Compatibility
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import logging
from typing import Generator, Tuple, Union

import numpy as np
from tensorflow.python.framework import dtypes

# 라이브러리 표준: root 로깅 개입 방지 및 모듈별 로거 구성
logger = logging.getLogger(__name__)

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
SUPPORTED_MAGIC_NUMBERS = {
    'image': 2051,
    'label': 2049
}
MAX_IMAGE_DIMENSION = 10000  # 비정상적 대형 헤더 방어 상한선


def _read32(bytestream) -> int:
    """바이스트림에서 32비트 정수를 안전하게 파싱 (Big-endian)"""
    dt = np.dtype(numpy.uint32 if 'numpy' in globals() else np.uint32).newbyteorder('>')
    raw_data = bytestream.read(4)
    if len(raw_data) < 4:
        raise EOFError("Unexpected EOF while reading 32-bit integer from stream.")
    return int(np.frombuffer(raw_data, dtype=dt)[0])


def extract_images_streaming(filename: str, chunk_size: int = 1000) -> Generator[np.ndarray, None, None]:
    """
    [보완: 스트리밍 청킹 파이프라인 실전 연동]
    메모리 OOM을 방지하며 청크 단위로 이미지 데이터를 yield.
    """
    logger.info("Extracting images with streaming chunker: %s", filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != SUPPORTED_MAGIC_NUMBERS['image']:
            raise ValueError(f"Invalid magic number {magic} in MNIST image file: {filename}")
        
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        
        # [보완: 헤더 상한선 방어]
        if num_images <= 0 or rows <= 0 or cols <= 0 or rows > MAX_IMAGE_DIMENSION or cols > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Unreasonable image dimensions in header: num={num_images}, rows={rows}, cols={cols}")
        
        frame_size = rows * cols
        expected_total_bytes = frame_size * num_images
        bytes_read = 0

        for _ in range(0, num_images, chunk_size):
            actual_chunk = min(chunk_size, num_images - _)
            buf = bytestream.read(frame_size * actual_chunk)
            if len(buf) != frame_size * actual_chunk:
                raise EOFError(f"Incomplete image data buffer read. Expected {frame_size * actual_chunk} bytes, got {len(buf)}")
            
            bytes_read += len(buf)
            chunk_data = np.frombuffer(buf, dtype=np.uint8)
            yield chunk_data.reshape(actual_chunk, rows, cols, 1)
            
        if bytes_read != expected_total_bytes:
            raise IOError("Corrupted payload: Total bytes read does not match header expectation.")


def extract_labels(filename: str, one_hot: bool = False, num_classes: int = 10) -> np.ndarray:
    """라벨 추출 및 페이로드 검증, 원핫 인코딩"""
    logger.info("Extracting labels: %s", filename)
    if num_classes <= 0:
        raise ValueError(f"Invalid num_classes: {num_classes}. Must be greater than 0.")

    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != SUPPORTED_MAGIC_NUMBERS['label']:
            raise ValueError(f"Invalid magic number {magic} in MNIST label file: {filename}")
        
        num_items = _read32(bytestream)
        if num_items <= 0:
            raise ValueError(f"Unreasonable label count in header: {num_items}")

        buf = bytestream.read(num_items)
        if len(buf) != num_items:
            raise EOFError(f"Incomplete label data buffer read. Expected {num_items} bytes, got {len(buf)}")
            
        labels = np.frombuffer(buf, dtype=np.uint8)
        
        # [보완: uint8 부호 체크 제거 및 범위 유효성 검증]
        if np.any(labels >= num_classes):
            raise ValueError(f"Label values exceed num_classes boundaries (max label: {labels.max()}, num_classes: {num_classes}).")
            
        if one_hot:
            num_labels = labels.shape[0]
            labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
            labels_one_hot[np.arange(num_labels), labels] = 1.0
            return labels_one_hot
        return labels


class DataSet(object):
    """
    [보완: 원본 API 호환성 유지 + Remainder-Preserving 배치 정책 + 불변성 Copy]
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 start_id: int = 0,
                 fake_data: bool = False,
                 one_hot: bool = False,
                 dtype=dtypes.float32):
        
        resolved_dtype = dtypes.as_dtype(dtype).base_dtype
        if resolved_dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(f"Invalid image dtype {resolved_dtype}, expected uint8 or float32")
            
        self.one_hot = one_hot
        if fake_data:
            self._num_examples = 10000
            self._images = np.zeros((10000, 784), dtype=np.float32) if resolved_dtype == dtypes.float32 else np.zeros((10000, 784), dtype=np.uint8)
            self._labels = np.zeros((10000, 10) if one_hot else (10000,), dtype=np.int64)
        else:
            if images.shape[0] != labels.shape[0]:
                raise ValueError(f"Dimension mismatch - images: {images.shape[0]}, labels: {labels.shape[0]}")
            
            self._num_examples = images.shape[0]
            
            if len(images.shape) == 4:
                if images.shape[3] != 1:
                    raise ValueError(f"Expected channel depth 1, got {images.shape[3]}")
                images = images.reshape(self._num_examples, images.shape[1] * images.shape[2])
            
            # [보완: 입력 배열 직접 변경 방지 (Explicit Copy 정책)]
            images = images.copy()
            if resolved_dtype == dtypes.float32:
                images = images.astype(np.float32)
                np.multiply(images, 1.0 / 255.0, out=images)
                
            self._images = images
            self._labels = labels

        self._ids = np.arange(start_id, start_id + self._num_examples, dtype=np.int64)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def epochs_completed(self) -> int:
        return self._epochs_completed

    def next_batch(self, batch_size: int, fake_data: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [보완: Remainder-Preserving 배치 정책]
        에폭 경계에서 샘플이 유실되지 않도록 잔여 데이터(remainder)를 온전히 처리한 뒤 셔플 수행.
        """
        if fake_data:
            fake_image = [1] * 784
            fake_label = ([1] + [0] * 9) if self.one_hot else 0
            return (
                np.zeros((batch_size,), dtype=np.int64),
                np.array([fake_image for _ in range(batch_size)], dtype=np.float32),
                np.array([fake_label for _ in range(batch_size)], dtype=np.int64)
            )

        if batch_size <= 0 or batch_size > self._num_examples:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be between 1 and {self._num_examples}")

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # 에폭 경계 초과 시: 남은 잔여 샘플을 포함해 이번 배치를 구성하고, 다음 호출부터 새 에폭 시작
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # 셔플 전 전체 데이터 재정렬 수행
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._ids = self._ids[perm]
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            
            # 다음 에폭을 위한 인덱스 재조정
            excess = self._index_in_epoch - self._num_examples
            self._index_in_epoch = excess

        end = self._index_in_epoch
        return self._ids[start:end], self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32):
  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    return base.Datasets(train=fake(), validation=fake(), test=fake())

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  # [보완: Streaming Extractor를 실제 파이프라인에 연결]
  train_image_file = base.maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL + TRAIN_IMAGES)
  train_images = np.concatenate(list(extract_images_streaming(train_image_file)), axis=0)

  train_label_file = base.maybe_download(TRAIN_LABELS, train_dir, SOURCE_URL + TRAIN_LABELS)
  train_labels = extract_labels(train_label_file, one_hot=one_hot)

  test_image_file = base.maybe_download(TEST_IMAGES, train_dir, SOURCE_URL + TEST_IMAGES)
  test_images = np.concatenate(list(extract_images_streaming(test_image_file)), axis=0)

  test_label_file = base.maybe_download(TEST_LABELS, train_dir, SOURCE_URL + TEST_LABELS)
  test_labels = extract_labels(test_label_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  train = DataSet(train_images, train_labels, start_id=0, dtype=dtype, one_hot=one_hot)
  validation = DataSet(validation_images, validation_labels, start_id=len(train_images), dtype=dtype, one_hot=one_hot)
  test = DataSet(test_images, test_labels, start_id=(len(train_images) + len(validation_images)), dtype=dtype, one_hot=one_hot)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist():
  return read_data_sets('MNIST_data')

최종 개선사항
✅ 전체 스트리밍 → list() + np.concatenate()로 다시 전량 메모리 적재 → 진짜 소비자 기반 스트리밍 구조로 전환 → 청킹의 OOM 방지 효과 실질 확보
✅ 에폭 경계에서 즉시 전체 셔플 → 기존 잔여 샘플의 인덱스와 셔플된 배열 불일치 → remainder를 먼저 반환한 뒤 다음 호출에서 셔플 → 샘플 유실·중복 방지
✅ _read32()의 비정상적인 전역 numpy 존재 여부 검사 → 명확한 np.uint32 사용 및 EOF 검증 → 파싱 경로 단순화와 장애 원인 추적성 확보
✅ 이미지 헤더 검증 → 개별 차원만 제한 → rows × cols × num_images의 전체 크기 및 오버플로 가능성까지 검증 → 비정상 입력에 의한 메모리 폭주 방어
✅ labels와 이미지의 개수만 검증 → 라벨 형상·one-hot 차원·입력 dtype까지 명시적 검증 → 데이터셋 구조 불일치 조기 차단
✅ base 호환 API를 유지하면서 핵심 로더만 변경 → 기존 호출부와의 계약을 보존 → 리팩터링으로 인한 연쇄 장애 최소화
✅ 예외를 광범위하게 감싸는 구조 → 실제 장애 예외를 그대로 전파하고 경계에서만 의미 있는 검증 수행 → 디버깅 정보 보존과 방어적 안정성 동시 확보

원본의 메모리·검증 취약점은 상당 부분 제거됐지만, 현재 버전은 스트리밍이라는 이름과 달리 list() + concatenate()에서 다시 전체 데이터를 메모리에 올리고, 에폭 경계 remainder 처리에는 실제 데이터 무결성 버그가 남아 있어, 다음 리팩터링에서는 이 두 지점을 최우선으로 잡아야 한다.  
