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

"""This module contains the data augmentation processes for MindMeld."""

import logging
import re

from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Dict, Type, Iterable

from ._util import get_pattern, read_path_queries, write_to_file
from .components._util import _is_module_available
from .markup import load_query
from .resource_loader import ResourceLoader

logger = logging.getLogger(__name__)

# pylint: disable=R0201

SUPPORTED_LANGUAGE_CODES = ["en", "es", "fr", "it", "pt", "ro"]


class UnsupportedLanguageError(Exception):
    pass


class AugmentorFactory:
    """Creates an Augmentor object.

    Attributes:
        config (dict): A model configuration.
        language (str): Language for data augmentation.
        resource_loader (object): Resource Loader object for the application.
    """

    def __init__(self, config, language, resource_loader):
        self.config = config
        self.language = language
        self.resource_loader = resource_loader

    def create_augmentor(self):
        """Creates an augmentor instance using the provided configuration

        Returns:
            Augmentor: An Augmentor class

        Raises:
            ValueError: When model configuration is invalid or required key is missing
        """
        if "augmentor_class" not in self.config:
            raise KeyError("Missing required argument in AUGMENTATION_CONFIG: 'augmentor_class'")

        # Validate configuration input
        batch_size = self.config.get("batch_size", 8)
        paths = self.config.get(
            "paths",
            [
                {
                    "domains": ".*",
                    "intents": ".*",
                    "files": ".*",
                }
            ],
        )
        path_suffix = self.config.get("path_suffix", "-augment.txt")
        retain_entities = self.config.get("retain_entities", False)
        try:
            return AUGMENTATION_MAP[self.config["augmentor_class"]](
                batch_size=batch_size,
                language=self.language,
                retain_entities=retain_entities,
                paths=paths,
                path_suffix=path_suffix,
                resource_loader=self.resource_loader,
            )
        except KeyError as e:
            msg = "Invalid model configuration: Unknown model type {!r}"
            raise ValueError(msg.format(self.config["augmentor_class"])) from e


class Augmentor(ABC):
    """
    Abstract Augmentor class.
    """

    def __init__(
        self, language: str, paths: Iterable[str], path_suffix: str, resource_loader: ResourceLoader
    ):
        """Initializes an augmentor.

        Args:
            language (str): The language code for paraphrasing
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        self.language_code = language
        self.files_to_augment = paths
        self.path_suffix = path_suffix
        self._resource_loader = resource_loader
        self._check_dependencies()
        self._check_language_support()

    def _check_dependencies(self):
        """Checks module dependencies."""
        if not _is_module_available("torch"):
            raise ModuleNotFoundError(
                "Library not found: 'torch'. Run 'pip install mindmeld[augment]' to install."
            )

        if not _is_module_available("transformers"):
            raise ModuleNotFoundError(
                "Library not found: 'transformers'. Run 'pip install mindmeld[augment]' to install."
            )

    def _check_language_support(self):
        """Checks if language is currently supported for augmentation."""
        if self.language_code not in SUPPORTED_LANGUAGE_CODES:
            raise UnsupportedLanguageError(
                f"'{self.language_code}' is not supported yet. "
                "English (en), French (fr), and Italian (it), Portuguese (pt), Romanian (ro) "
                " and Spanish (es) are currently supported."
            )

    def augment(self, **kwargs):
        """Augments queries given initial queries in application."""
        filtered_paths = self._get_files(path_rules=self.files_to_augment)

        for path in tqdm(filtered_paths):
            queries = self._get_processed_queries_to_paraphrase(path)
            # To-Do: Use generator to write files incrementally.
            augmented_queries = self.augment_queries(queries, **kwargs)
            write_to_file(path, augmented_queries, suffix=self.path_suffix)

    @abstractmethod
    def augment_queries(self, queries):
        """Generates augmented data given application queries.

        Args:
            queries (list): List of queries.

        Return:
            augmented_queries (list): List of augmented queries.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _prepare_inputs(self, queries):
        """Prepare data to be fed to the models as input

        Args:
            queries (list(str)): List of queries to be paraphrased

        Returns:
            formatted queries (list(str))

        """
        raise NotImplementedError("Subclasses must implement this method")

    def _validate_generated_query(self, query: str):
        """Validates whether augmented query has atleast one alphanumeric character

        Args:
            query (str): Generated query to be validated.
        """
        pattern = re.compile(r"^.*[a-zA-Z0-9].*$")
        return pattern.search(query) and True

    def _get_processed_queries_to_paraphrase(self, path):
        """Returns a list of processed queries for a given file path

        Args:
            path (str): Path to text file with queries

        Return:
            Processed queries (list(ProcessedQuery))
        """
        queries = read_path_queries(path)
        processed_queries = []
        for query in queries:
            processed_query = load_query(query, query_factory=self._resource_loader.query_factory)
            processed_queries.append(processed_query)
        return processed_queries

    def _get_files(self, path_rules=None):
        """Fetches relevant files given the path rules specified in the config.

        Args:
            path_rules (list): Path rules for fetching relevant files.

        Return:
            filtered_paths (list): List of file paths to be augmeted.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()

        if not path_rules:
            logger.warning(
                """'paths' field is not configured or misconfigured in the `config.py`.
                 Can't find files to augment."""
            )
            return []

        filtered_paths = []

        for rule in path_rules:
            pattern = get_pattern(rule)
            compiled_pattern = re.compile(pattern)
            filtered_paths.extend(
                self._resource_loader.filter_file_paths(
                    compiled_pattern=compiled_pattern, file_paths=all_file_paths
                )
            )
        return filtered_paths


AUGMENTATION_MAP: Dict[str, Type[Augmentor]] = {}


def register_augmentor(augmentor_name: str, augmentor_class: Type[Augmentor]):
    """Registers an Augmentor class for use with `create_augmentor()`

    Args:
        annotator_class_name (str): The annotator class name as specified in the config
        model_class (class): The annotator class to register
    """
    AUGMENTATION_MAP[augmentor_name] = augmentor_class
