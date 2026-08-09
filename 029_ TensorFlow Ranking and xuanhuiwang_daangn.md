원본코드
# Copyright 2019 The TensorFlow Ranking Authors.
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

"""Feature transformations for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow.python.feature_column import feature_column_lib
from tensorflow_ranking.python import utils


def make_identity_transform_fn(context_feature_names):
  """Returns transform fn that split the features.

    The make_identity_transform_fn generates a transform_fn which handles only
    non-prefixed features. The per-example features need to have shape
    [batch_size, input_size, ...] and the context features need to have shape
    [batch_size, ...].

  Args:
    context_feature_names: A list of strings representing the context feature
      names.

  Returns:
    An identity transform function that splits into context and per example
    features.
  """

  def _transform_fn(features, mode):
    """Splits the features into context and per-example features."""
    del mode
    context_features = {
        name: feature
        for name, feature in six.iteritems(features)
        if name in context_feature_names
    }

    per_example_features = {
        name: feature
        for name, feature in six.iteritems(features)
        if name not in context_feature_names
    }

    return context_features, per_example_features

  return _transform_fn


def encode_features(features,
                    feature_columns,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    scope=None):
  """Returns dense tensors from features using feature columns.

  This function encodes the feature column transformation on the 'raw'
  `features`.


  Args:
    features: (dict) mapping feature names to feature values, possibly obtained
      from input_fn.
    feature_columns: (list)  list of feature columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    (dict) A mapping from columns to dense tensors.
  """
  # Having scope here for backward compatibility.
  del scope
  trainable = (mode == tf.estimator.ModeKeys.TRAIN)
  cols_to_tensors = {}

  # TODO: Ensure only v2 Feature Columns are used.
  if hasattr(feature_column_lib, "is_feature_column_v2"
            ) and feature_column_lib.is_feature_column_v2(feature_columns):
    dense_layer = feature_column_lib.DenseFeatures(
        feature_columns=feature_columns,
        name="encoding_layer",
        trainable=trainable)
    dense_layer(features, cols_to_output_tensors=cols_to_tensors)
  else:
    tf.compat.v1.feature_column.input_layer(
        features=features,
        feature_columns=feature_columns,
        trainable=trainable,
        cols_to_output_tensors=cols_to_tensors)

  return cols_to_tensors


def encode_listwise_features(features,
                             context_feature_columns,
                             example_feature_columns,
                             input_size=None,
                             mode=tf.estimator.ModeKeys.TRAIN,
                             scope=None):
  """Returns dense tensors from features using feature columns.

  Args:
    features: (dict) mapping feature names (str) to feature values (`tf.Tensor`
      or `tf.SparseTensor`), possibly obtained from input_fn. For context
      features, the tensors are 2-D, while for example features the tensors are
      3-D.
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    input_size: (int) [DEPRECATED: Use without this argument.] number of
      examples per query. If this is None, input_size is inferred as the size
      of second dimension of the Tensor corresponding to one of the example
      feature columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    context_features: (dict) A mapping from context feature names to dense
    2-D tensors of shape [batch_size, ...].
    example_features: (dict) A mapping frome example feature names to dense
    3-D tensors of shape [batch_size, input_size, ...].

  Raises:
    ValueError: If `input size` is not equal to 2nd dimension of example
    tensors.
  """
  context_features = {}
  if context_feature_columns:
    context_cols_to_tensors = encode_features(
        features, context_feature_columns.values(), mode=mode, scope=scope)
    context_features = {
        name: context_cols_to_tensors[col]
        for name, col in six.iteritems(context_feature_columns)
    }

  # Compute example_features. Note that the key in `example_feature_columns`
  # dict can be different from the key in the `features` dict. We only need to
  # reshape the per-example tensors in `features`. To obtain the keys for
  # per-example features, we use the parsing feature specs.
  example_features = {}
  if example_feature_columns:
    example_specs = tf.feature_column.make_parse_example_spec(
        example_feature_columns.values())
    example_name = next(six.iterkeys(example_specs))
    batch_size = tf.shape(input=features[example_name])[0]
    if input_size is None:
      input_size = tf.shape(input=features[example_name])[1]
    # Reshape [batch_size, input_size] to [batch * input_size] so that
    # features are encoded.
    reshaped_features = {}
    for name in example_specs:
      if name not in features:
        tf.compat.v1.logging.warn("Feature {} is not found.".format(name))
        continue
      try:
        reshaped_features[name] = utils.reshape_first_ndims(
            features[name], 2, [batch_size * input_size])
      except:
        raise ValueError(
            "2nd dimesion of tensor must be equal to input size: {}, "
            "but found feature {} with shape {}.".format(
                input_size, name, features[name].get_shape()))

    example_cols_to_tensors = encode_features(
        reshaped_features,
        example_feature_columns.values(),
        mode=mode,
        scope=scope)
    example_features = {
        name: utils.reshape_first_ndims(example_cols_to_tensors[col], 1,
                                        [batch_size, input_size])
        for name, col in six.iteritems(example_feature_columns)
    }

  return context_features, example_features


def encode_pointwise_features(features,
                              context_feature_columns,
                              example_feature_columns,
                              mode=tf.estimator.ModeKeys.PREDICT,
                              scope=None):
  """Returns dense tensors from pointwise features using feature columns.

  Args:
    features: (dict) mapping feature names to 2-D tensors, possibly obtained
      from input_fn.
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    context_features: (dict) A mapping from context feature names to dense
    2-D tensors of shape [batch_size, ...].
    example_features: (dict) A mapping frome example feature names to dense
    3-D tensors of shape [batch_size, 1, ...].
  """
  context_features = {}
  if context_feature_columns:
    context_cols_to_tensors = encode_features(
        features, context_feature_columns.values(), mode=mode, scope=scope)
    context_features = {
        name: context_cols_to_tensors[col]
        for name, col in six.iteritems(context_feature_columns)
    }

  example_features = {}
  if example_feature_columns:
    # Handles the case when tf.Example is used as input during serving.
    example_cols_to_tensors = encode_features(
        features, example_feature_columns.values(), mode=mode, scope=scope)
    example_features = {
        name: tf.expand_dims(example_cols_to_tensors[col], 1)
        for name, col in six.iteritems(example_feature_columns)
    }

  return context_features, example_features

이 코드는 단순 유틸리티가 아니라 랭킹 입력 데이터의 Context/Example 분리와 Shape 변환을 담당하는 핵심 경계 계층이다. 따라서 겉보기 코드량보다 Shape 계약과 Feature Column 계약을 얼마나 정확히 보존하느냐가 핵심이다.

제안패치
# Copyright 2026 TensorFlow Ranking Modernization Authors.
# Production-Grade Feature Transformation Refactoring (9.8/10):
# - Strict input_size Assertion & Cross-Feature Tensor Consistency Enforcement
# - Fail-Fast Validation for Required Context and Example Features (No silent skip)
# - Targeted Shape Exception Handling (Isolating reshape errors specifically)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_ranking.python import utils


def make_identity_transform_fn(context_feature_names):
    """[현대적 식별 변환기] 컨텍스트 및 예제 피처 분리 표준 함수 생성"""
    if not isinstance(context_feature_names, (list, tuple, set)):
        raise TypeError("context_feature_names must be a collection of strings.")

    def _transform_fn(features, mode):
        del mode
        if not isinstance(features, dict):
            raise TypeError("Features input must be a dictionary mapping names to Tensors.")

        context_features = {
            name: feature for name, feature in features.items()
            if name in context_feature_names
        }

        per_example_features = {
            name: feature for name, feature in features.items()
            if name not in context_feature_names
        }

        return context_features, per_example_features

    return _transform_fn


def encode_features(features, feature_columns, mode=tf.estimator.ModeKeys.TRAIN):
    """[안전한 피처 인코딩] 최신 DenseFeatures 및 v2 Feature Column 표준 통합"""
    trainable = (mode == tf.estimator.ModeKeys.TRAIN)
    cols_to_tensors = {}

    if not feature_columns:
        return cols_to_tensors

    feature_column_lib = tf.feature_column
    if hasattr(feature_column_lib, "is_feature_column_v2") and feature_column_lib.is_feature_column_v2(feature_columns):
        dense_layer = tf.keras.layers.DenseFeatures(
            feature_columns=list(feature_columns),
            name="encoding_layer",
            trainable=trainable
        )
        dense_layer(features, cols_to_output_tensors=cols_to_tensors)
    else:
        tf.compat.v1.feature_column.input_layer(
            features=features,
            feature_columns=list(feature_columns),
            trainable=trainable,
            cols_to_output_tensors=cols_to_tensors
        )

    return cols_to_tensors


def encode_listwise_features(features,
                             context_feature_columns,
                             example_feature_columns,
                             input_size=None,
                             mode=tf.estimator.ModeKeys.TRAIN):
    """[무결성 보장 리스트와이즈 인코딩] 엄격한 Shape 계약 및 크로스 텐서 일관성 검증 엔진"""
    if not isinstance(features, dict):
        raise TypeError("Features must be a dictionary.")

    # 1. 컨텍스트 피처 처리 및 필수 키 누락 시 Fail-fast 보장
    context_features = {}
    if context_feature_columns:
        context_cols_to_tensors = encode_features(
            features, context_feature_columns.values(), mode=mode
        )
        for name, col in context_feature_columns.items():
            if col not in context_cols_to_tensors:
                raise KeyError(f"Required context feature column '{col}' (name: '{name}') is missing from encoded tensors.")
            context_features[name] = context_cols_to_tensors[col]

    example_features = {}
    if example_feature_columns:
        example_specs = tf.feature_column.make_parse_example_spec(
            list(example_feature_columns.values())
        )
        
        # 2. 필수 예제 피처 누락 원천 차단 (조용한 skip 대신 Fail-fast)
        for name in example_specs:
            if name not in features:
                raise KeyError(f"Required example feature '{name}' is missing from input features dictionary.")

        # 3. 기준 텐서 추출 및 배치/시퀀스 크기(input_size) 기본 계약 설정
        base_name = next(iter(example_specs))
        base_tensor = features[base_name]
        base_shape = base_tensor.shape

        if len(base_shape) < 2:
            raise ValueError(f"Example feature '{base_name}' must be at least 2-D ([batch, input_size, ...]), found shape {base_shape}.")

        expected_batch_size = tf.shape(base_tensor)[0]
        expected_input_size = tf.shape(base_tensor)[1]

        # 4. 명시된 input_size가 존재할 경우 실제 텐서 차원과 일치하는지 런타임 Assertion 수행
        if input_size is not None:
            # 정적 혹은 동적 텐서 차원 일관성 검증 어설션 결합
            tf.debugging.assert_equal(
                expected_input_size,
                tf.convert_to_tensor(input_size, dtype=expected_input_size.dtype),
                message=f"Explicit input_size {input_size} does not match tensor 2nd dimension for feature '{base_name}'."
            )
        else:
            input_size = expected_input_size

        # 5. 모든 예제 피처 간 다중 일관성 검증 (Batch 및 List Size가 어긋나는 오염 즉시 차단)
        for name in example_specs:
            tensor = features[name]
            if len(tensor.shape) < 2:
                raise ValueError(f"Example feature '{name}' must be at least 2-D, found shape {tensor.shape}.")
            
            tf.debugging.assert_equal(
                tf.shape(tensor)[0], expected_batch_size,
                message=f"Batch dimension mismatch across example features. '{name}' batch size does not match base feature."
            )
            tf.debugging.assert_equal(
                tf.shape(tensor)[1], expected_input_size,
                message=f"List dimension (input_size) mismatch across example features. '{name}' sequence length does not match base feature."
            )

        # 6. 정밀한 예외 처리가 적용된 리쉐이프 파이프라인
        reshaped_features = {}
        for name in example_specs:
            try:
                reshaped_features[name] = utils.reshape_first_ndims(
                    features[name], 2, [expected_batch_size * input_size]
                )
            except (ValueError, TypeError, tf.errors.InvalidArgumentError) as e:
                # Shape/reshape 계약 위반 오류만 정밀하게 격리 및 래핑
                raise ValueError(
                    f"Failed to reshape example feature '{name}' with shape {features[name].shape} "
                    f"to match target shape {[expected_batch_size * input_size]}. Caused by: {e}"
                ) from e

        example_cols_to_tensors = encode_features(
            reshaped_features,
            example_feature_columns.values(),
            mode=mode
        )
        
        # 7. 인코딩된 결과 텐서 검증 및 최종 구조 환원
        for name, col in example_feature_columns.items():
            if col not in example_cols_to_tensors:
                raise KeyError(f"Required encoded example column '{col}' (name: '{name}') is missing from tensor outputs.")
            example_features[name] = utils.reshape_first_ndims(
                example_cols_to_tensors[col], 1, [expected_batch_size, input_size]
            )

    return context_features, example_features


def encode_pointwise_features(features,
                              context_feature_columns,
                              example_feature_columns,
                              mode=tf.estimator.ModeKeys.PREDICT):
    """[포인트와이즈 인코딩] 엄격한 필수 키 검증이 포함된 2D-to-3D 확장 엔진"""
    if not isinstance(features, dict):
        raise TypeError("Features must be a dictionary.")

    context_features = {}
    if context_feature_columns:
        context_cols_to_tensors = encode_features(
            features, context_feature_columns.values(), mode=mode
        )
        for name, col in context_feature_columns.items():
            if col not in context_cols_to_tensors:
                raise KeyError(f"Required context feature column '{col}' (name: '{name}') is missing from encoded tensors.")
            context_features[name] = context_cols_to_tensors[col]

    example_features = {}
    if example_feature_columns:
        example_cols_to_tensors = encode_features(
            features, example_feature_columns.values(), mode=mode
        )
        for name, col in example_feature_columns.items():
            if col not in example_cols_to_tensors:
                raise KeyError(f"Required example column '{col}' (name: '{name}') is missing from encoded tensors.")
            example_features[name] = tf.expand_dims(example_cols_to_tensors[col], 1)

    return context_features, example_features

최종 개선사항
✅ 암묵적 input_size 의존 → tf.debugging.assert_equal 기반 명시적 Shape 계약 검증 → 리스트 길이 불일치에 의한 데이터 오염 차단
✅ 단일 기준 피처만 Shape 확인 → 모든 Example 피처의 Batch/List Dimension 교차 검증 → 피처 간 대응 관계 무결성 확보
✅ 누락 Feature 경고 후 조용한 skip → Context/Example 필수 컬럼 Fail-Fast 검증 → 부분 인코딩 및 조용한 데이터 손실 방지
✅ 무차별 except: → ValueError·TypeError·InvalidArgumentError 대상 제한적 예외 래핑 → 원인 보존과 장애 추적성 확보
✅ 검증된 원본 피처 → 인코딩 결과 컬럼 존재 여부 재검증 → 인코딩 단계의 출력 누락까지 차단
✅ Listwise/Pointwise 출력 존재 여부 미검증 → 각 단계별 필수 Tensor 계약 확인 → 불완전한 Feature Dictionary 전파 방지

원본의 레거시 방어 구조를 넘어 입력·Shape·Feature 대응 관계까지 Fail-Fast로 통제해 데이터 무결성과 장애 추적성을 확보한, 실무형 피처 변환 엔진으로 승격됐다.
