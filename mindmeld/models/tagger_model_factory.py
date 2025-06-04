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

"""This module contains the Memm entity recognizer."""
import logging

from .model import ModelConfig
from .tagger_models import TaggerModel, PytorchTaggerModel
from .model import AbstractModelFactory

logger = logging.getLogger(__name__)


class TaggerModelFactory(AbstractModelFactory):
    @staticmethod
    def get_model_cls(config: ModelConfig):
        CLASSES = [TaggerModel, PytorchTaggerModel]
        classifier_type = config.model_settings["classifier_type"]

        for _class in CLASSES:
            if classifier_type in _class.ALLOWED_CLASSIFIER_TYPES:
                return _class

        msg = (
            f"Invalid 'classifier_type': {classifier_type}. "
            f"Allowed types are: {[_class.ALLOWED_CLASSIFIER_TYPES for _class in CLASSES]}"
        )
        raise ValueError(msg)
