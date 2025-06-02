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

"""This module contains base classes for models defined in the models subpackage."""

import json
import logging
from typing import (
    Dict,
    Tuple,
    Pattern,
    Set,
)

from .helpers import (
    CHAR_NGRAM_FREQ_RSC,
    WORD_NGRAM_FREQ_RSC,
    get_feature_extractor,
)

logger = logging.getLogger(__name__)


class ModelConfig:
    """A value object representing a model configuration.

    Attributes:
        model_type (str): The name of the model type. Will be used to find the
            model class to instantiate
        example_type (str): The type of the examples which will be passed into
            `fit()` and `predict()`. Used to select feature extractors
        label_type (str): The type of the labels which will be passed into
            `fit()` and returned by `predict()`. Used to select the label encoder
        model_settings (dict): Settings specific to the model type specified
        params (dict): Params to pass to the underlying classifier
        param_selection (dict): Configuration for param selection (using cross
            validation)
            {'type': 'shuffle',
            'n': 3,
            'k': 10,
            'n_jobs': 2,
            'scoring': '',
            'grid': {}
            }
        features (dict): The keys are the names of feature extractors and the
            values are either a kwargs dict which will be passed into the
            feature extractor function, or a callable which will be used as to
            extract features
        train_label_set (regex pattern): The regex pattern for finding training
            file names.
        test_label_set (regex pattern): The regex pattern for finding testing
            file names.
    """

    __slots__ = [
        "model_type",
        "example_type",
        "label_type",
        "features",
        "model_settings",
        "params",
        "param_selection",
        "train_label_set",
        "test_label_set",
    ]

    def __init__(
        self,
        model_type: str = None,
        example_type: str = None,
        label_type: str = None,
        features: Dict = None,
        model_settings: Dict = None,
        params: Dict = None,
        param_selection: Dict = None,
        train_label_set: Pattern[str] = None,
        test_label_set: Pattern[str] = None,
    ):
        for arg, val in {
            "model_type": model_type,
            "label_type": label_type,
        }.items():
            if val is None:
                raise TypeError("__init__() missing required argument {!r}".format(arg))
        self.model_type = model_type
        self.example_type = example_type
        self.label_type = label_type
        self.features = features
        self.model_settings = model_settings
        self.params = params
        self.param_selection = param_selection
        self.train_label_set = train_label_set
        self.test_label_set = test_label_set

    def __repr__(self):
        args_str = ", ".join("{}={!r}".format(key, getattr(self, key)) for key in self.__slots__)
        return "{}({})".format(self.__class__.__name__, args_str)

    def to_dict(self) -> Dict:
        """Converts the model config object into a dict

        Returns:
            dict: A dict version of the config
        """
        result = {}
        for attr in self.__slots__:
            result[attr] = getattr(self, attr)
        return result

    def to_json(self) -> str:
        """Converts the model config object to JSON

        Returns:
            str: JSON representation of the classifier
        """
        return json.dumps(self.to_dict(), sort_keys=True)

    def resolve_config(self, new_config: "ModelConfig"):
        """This method resolves any config incompatibility issues by
        loading the latest settings from the app config to the current config

        Args:
            new_config (ModelConfig): The ModelConfig representing the app's latest config
        """
        new_settings = ["train_label_set", "test_label_set"]
        logger.warning(
            "Loading missing properties %s from app " "configuration file",
            new_settings,
        )
        for setting in new_settings:
            setattr(self, setting, getattr(new_config, setting))

    def get_ngram_lengths_and_thresholds(self, rname: str) -> Tuple:
        """
        Returns the n-gram lengths and thresholds to extract to optimize resource collection

        Args:
            rname (string): Name of the resource

        Returns:
            (tuple): tuple containing:

                * lengths (list of int): list of n-gram lengths to be extracted
                * thresholds (list of int): thresholds to be applied to corresponding n-gram lengths
        """
        lengths = thresholds = None
        # if it's not the n-gram feature, we don't need length and threshold information
        if rname == CHAR_NGRAM_FREQ_RSC:
            feature_name = "char-ngrams"
        elif rname == WORD_NGRAM_FREQ_RSC:
            feature_name = "bag-of-words"
        else:
            return lengths, thresholds

        # feature name varies based on whether it's for a classifier or tagger
        if self.model_type == "text":
            if feature_name in self.features:
                lengths = self.features[feature_name]["lengths"]
                thresholds = self.features[feature_name].get("thresholds", [1] * len(lengths))
        elif self.model_type == "tagger":
            feature_name = feature_name + "-seq"
            if feature_name in self.features:
                lengths = self.features[feature_name]["ngram_lengths_to_start_positions"].keys()
                thresholds = self.features[feature_name].get("thresholds", [1] * len(lengths))

        return lengths, thresholds

    def required_resources(self) -> Set:
        """Returns the resources this model requires

        Returns:
            set: set of required resources for this model
        """
        # get list of resources required by feature extractors
        required_resources = set()
        if self.features:
            for name in self.features:
                feature = get_feature_extractor(self.example_type, name)
                required_resources.update(feature.__dict__.get("requirements", []))
        return required_resources
