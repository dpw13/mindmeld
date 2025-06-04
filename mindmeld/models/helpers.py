"""This module contains some helper functions for the models package"""

# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import re
from tempfile import mkstemp
from collections.abc import Callable
from typing import Any, Dict, Iterable, Tuple, Generator

import numpy as np

import nltk
from sklearn.metrics import make_scorer

from ..gazetteer import Gazetteer
from ..text_preparation.text_preparation_pipeline import (
    TextPreparationPipeline,
    TextPreparationPipelineFactory,
)

logger = logging.getLogger(__name__)

FEATURE_MAP: Dict[str, Dict[str, Callable]] = {}

# Example types
QUERY_EXAMPLE_TYPE = "query"
ENTITY_EXAMPLE_TYPE = "entity"

# resource/requirements names
GAZETTEER_RSC = "gazetteers"
QUERY_FREQ_RSC = "q_freq"
SYS_TYPES_RSC = "sys_types"
ENABLE_STEMMING = "enable-stemming"
WORD_FREQ_RSC = "w_freq"
WORD_NGRAM_FREQ_RSC = "w_ngram_freq"
CHAR_NGRAM_FREQ_RSC = "c_ngram_freq"
SENTIMENT_ANALYZER = "vader_classifier"
OUT_OF_BOUNDS_TOKEN = "<$>"
OUT_OF_VOCABULARY = "OOV"
IN_VOCABULARY = "IV"
DEFAULT_SYS_ENTITIES = [
    "sys_time",
    "sys_temperature",
    "sys_volume",
    "sys_amount-of-money",
    "sys_email",
    "sys_url",
    "sys_number",
    "sys_ordinal",
    "sys_duration",
    "sys_phone-number",
]


def get_feature_extractor(example_type: str, name: str) -> Callable:
    """Gets a feature extractor given the example type and name

    Args:
        example_type (str): The type of example
        name (str): The name of the feature extractor

    Returns:
        function: A feature extractor wrapper
    """
    return FEATURE_MAP[example_type][name]


def register_query_feature(feature_name: str) -> Callable:
    """Registers query feature

    Args:
        feature_name (str): The name of the query feature

    Returns:
        (func): the feature extractor
    """
    return register_feature(QUERY_EXAMPLE_TYPE, feature_name=feature_name)


def register_entity_feature(feature_name):
    """Registers entity feature

    Args:
        feature_name (str): The name of the entity feature

    Returns:
        (func): the feature extractor
    """
    return register_feature(ENTITY_EXAMPLE_TYPE, feature_name=feature_name)


def register_feature(feature_type: str, feature_name: str) -> Callable:
    """
    Decorator for adding feature extractor mappings to FEATURE_MAP

    Args:
        feature_type: 'query' or 'entity'
        feature_name: The name of the feature, used in config.py

    Returns:
        (func): the feature extractor
    """

    def add_feature(func):
        if feature_type not in {QUERY_EXAMPLE_TYPE, ENTITY_EXAMPLE_TYPE}:
            raise TypeError("Feature type can only be 'query' or 'entity'")

        # Add func to feature map with given type and name
        if feature_type in FEATURE_MAP:
            FEATURE_MAP[feature_type][feature_name] = func
        else:
            FEATURE_MAP[feature_type] = {feature_name: func}
        return func

    return add_feature


def mask_numerics(token: str) -> str:
    """Masks digit characters in a token

    Args:
        token (str): A string

    Returns:
        str: A masked string for digit characters
    """
    if token.isdigit():
        return "#NUM"
    else:
        return re.sub(r"\d", "8", token)


def get_ngram(tokens: Iterable[str], start: int, length: int) -> str:
    """Gets a ngram from a list of tokens.

    Handles out-of-bounds token positions with a special character.

    Args:
        tokens (list of str): Word tokens.
        start (int): The index of the desired ngram's start position.
        length (int): The length of the n-gram, e.g. 1 for unigram, etc.

    Returns:
        (str) An n-gram in the input token list.
    """

    ngram_tokens = []
    for index in range(start, start + length):
        token = OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens) else tokens[index]
        ngram_tokens.append(token)
    return " ".join(ngram_tokens)


def get_ngrams_upto_n(
    tokens: Iterable[str], n: int
) -> Generator[Tuple[Tuple, Tuple[int, int]], Any, None]:
    """This function returns a generator that returns ngram tuples with length upto n

    Args:
        tokens (list of str): Word tokens.
        n (int): The length of n-gram upto which the ngram tokens are generated

    Returns:
        tuple: ngram, (token index start, token index end)
    """
    if n == 0:
        return None
    for length, i in enumerate(range(1, n + 1)):
        for idx, j in enumerate(nltk.ngrams(tokens, i)):
            yield j, (idx, idx + length)

    return None


def get_seq_accuracy_scorer() -> Callable:
    """
    Returns a scorer that can be used by sklearn's GridSearchCV based on the
    sequence_accuracy_scoring method below.
    """
    return make_scorer(score_func=sequence_accuracy_scoring)


def get_seq_tag_accuracy_scorer() -> Callable:
    """
    Returns a scorer that can be used by sklearn's GridSearchCV based on the
    sequence_tag_accuracy_scoring method below.
    """
    return make_scorer(score_func=sequence_tag_accuracy_scoring)


def sequence_accuracy_scoring(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Accuracy score which calculates two sequences to be equal only if all of
        their predicted tags are equal.

    Args:
        y_true (list): A sequence of true expected labels
        y_pred (list): A sequence of predicted labels

    Returns:
        float: The sequence-level accuracy when comparing the predicted labels \
            against the true expected labels
    """
    total = len(y_true)
    if not total:
        return 0

    matches = sum(1 for yseq_true, yseq_pred in zip(y_true, y_pred) if yseq_true == yseq_pred)

    return float(matches) / float(total)


def sequence_tag_accuracy_scoring(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Accuracy score which calculates the number of tags that were predicted
        correctly.

    Args:
        y_true (list): A sequence of true expected labels
        y_pred (list): A sequence of predicted labels

    Returns:
        float: The tag-level accuracy when comparing the predicted labels \
            against the true expected labels
    """
    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]

    total = len(y_true_flat)
    if not total:
        return 0

    matches = sum(
        1 for (y_true_tag, y_pred_tag) in zip(y_true_flat, y_pred_flat) if y_true_tag == y_pred_tag
    )

    return float(matches) / float(total)


def entity_seqs_equal(expected: Iterable, predicted: Iterable) -> bool:
    """
    Returns true if the expected entities and predicted entities all match, returns
    false otherwise. Note that for entity comparison, we compare that the span, text,
    and type of all the entities match.

    Args:
        expected (list of core.Entity): A list of the expected entities for some query
        predicted (list of core.Entity): A list of the predicted entities for some query
    """
    if len(expected) != len(predicted):
        return False
    for expected_entity, predicted_entity in zip(expected, predicted):
        if expected_entity.entity.type != predicted_entity.entity.type:
            return False
        if expected_entity.span != predicted_entity.span:
            return False
        if expected_entity.text != predicted_entity.text:
            return False
    return True


def merge_gazetteer_resource(
    resource: Dict,
    dynamic_resource: Dict,
    text_preparation_pipeline: TextPreparationPipeline,
) -> Dict:
    """
    Returns a new resource that is a merge between the original resource and the dynamic
    resource passed in for only the gazetteer values

    Args:
        resource (dict): The original resource built from the app
        dynamic_resource (dict): The dynamic resource passed in
        text_preparation_pipeline (TextPreparationPipeline): For text tokenization and normalization

    Returns:
        dict: The merged resource
    """
    return_obj = {}
    for key in resource:
        # Pass by reference if not a gazetteer key
        if key != GAZETTEER_RSC:
            return_obj[key] = resource[key]
            continue

        # Create a dict from scratch if we match the gazetteer key
        return_obj[key] = {}
        for entity_type in resource[key]:
            # If the entity type is in the dyn gaz, we merge the data. Else,
            # just pass by reference the original resource data
            if entity_type in dynamic_resource[key]:
                new_gaz = Gazetteer(entity_type, text_preparation_pipeline)
                # We deep copy here since shallow copying will also change the
                # original resource's data during the 'update_entity' op.
                new_gaz.from_dict(resource[key][entity_type])

                for entity in dynamic_resource[key][entity_type]:
                    new_gaz.update_entity(
                        text_preparation_pipeline.normalize(entity),
                        dynamic_resource[key][entity_type][entity],
                    )

                # The new gaz created is a deep copied version of the merged gaz data
                return_obj[key][entity_type] = new_gaz.to_dict()
            else:
                return_obj[key][entity_type] = resource[key][entity_type]
    return return_obj


def ingest_dynamic_gazetteer(
    resource: Dict,
    dynamic_resource: Dict = None,
    text_preparation_pipeline: TextPreparationPipeline = None,
) -> Dict:
    """Ingests dynamic gazetteers from the app and adds them to the resource

    Args:
        resource (dict): The original resource
        dynamic_resource (dict, optional): The dynamic resource that needs to be ingested
        text_preparation_pipeline (TextPreparationPipeline): For text tokenization and normalization

    Returns:
        (dict): A new resource with the ingested dynamic resource
    """
    if not dynamic_resource or GAZETTEER_RSC not in dynamic_resource:
        return resource
    text_preparation_pipeline = (
        text_preparation_pipeline
        or TextPreparationPipelineFactory.create_default_text_preparation_pipeline()
    )
    workspace_resource = merge_gazetteer_resource(
        resource, dynamic_resource, text_preparation_pipeline
    )
    return workspace_resource


def requires(resource: str) -> Callable:
    """
    Decorator to enforce the resource dependencies of the active feature extractors

    Args:
        resource (str): the key of a classifier resource which must be initialized before
            the given feature extractor is used

    Returns:
        (func): the feature extractor
    """

    def add_resource(func):
        req = func.__dict__.get("requirements", set())
        req.add(resource)
        func.requirements = req
        return func

    return add_resource


def np_encoder(val):
    if isinstance(val, np.generic):
        return val.item()
    raise TypeError(f"{type(val)} cannot be serialized by JSON.")


class FileBackedList:
    """
    FileBackedList implements an interface for simple list use cases
    that is backed by a temporary file on disk.  This is useful for
    simple list processing in a memory efficient way.
    """

    def __init__(self):
        self.num_lines = 0
        self.file_handle = None
        fd, self.filename = mkstemp()
        os.close(fd)

    def __len__(self):
        return self.num_lines

    def append(self, line):
        if self.file_handle is None:
            # pylint: disable=consider-using-with
            self.file_handle = open(self.filename, "w")
        self.file_handle.write(json.dumps(line, default=np_encoder))
        self.file_handle.write("\n")
        self.num_lines += 1

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
        os.unlink(self.filename)

    def __iter__(self):
        # Flush out any remaining data to be written
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        return FileBackedList.Iterator(self)

    class Iterator:
        def __init__(self, source: "FileBackedList"):
            self.source = source
            # pylint: disable=consider-using-with
            self.file_handle = open(source.filename, "r")

        def __len__(self):
            return len(self.source)

        def __next__(self):
            try:
                line = next(self.file_handle)
                return json.loads(line)
            except Exception as e:
                self.file_handle.close()
                self.file_handle = None
                if not isinstance(e, StopIteration):
                    logger.error("Error reading from FileBackedList")
                raise

        def __del__(self):
            if self.file_handle:
                self.file_handle.close()
