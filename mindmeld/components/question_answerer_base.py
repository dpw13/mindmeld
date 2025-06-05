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

"""
This module contains the question answerer base classes of MindMeld.
"""
import copy
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Type

from ._config import (
    get_app_namespace,
    get_classifier_config,
)
from ._util import _is_module_available

logger = logging.getLogger(__name__)

DEFAULT_QUERY_TYPE = "keyword"
ALL_QUERY_TYPES = [
    "keyword",
    "text",
    "embedder",
    "embedder_keyword",
    "embedder_text",
]
EMBEDDING_FIELD_STRING = "_embedding"

QUESTION_ANSWERER_MODEL_MAPPINGS: Dict[str, "BaseQuestionAnswerer"] = {}


class QuestionAnswererFactory:
    """
    Factory class for creating QuestionAnswerers

    usage
        >>> question_answerer = QuestionAnswererFactory.create_question_answerer(**kwargs)
        >>> question_answerer.load_kb(...)
        >>> question_answerer.get(...) # .get(...) or .build_search(...)
    """

    @classmethod
    def create_question_answerer(cls, app_path=None, config=None, app_namespace=None, **kwargs):
        """
        Args:
            app_path (str, optional): The path to the directory containing the app's data. If
                provided, used to obtain default 'app_namespace' and QA configurations
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.
            config (dict, optional): The QA config if passed directly rather than loaded from the
                app config
        """

        config = cls._get_config(config, app_path)
        reformatted_config = cls._correct_deprecated_qa_config(config)

        model_type = reformatted_config.get("model_type")
        question_answerer_class = cls._get_question_answerer_class(model_type)

        return question_answerer_class(
            app_path=app_path,
            config=reformatted_config,
            app_namespace=app_namespace,
            **kwargs,
        )

    @staticmethod
    def _get_config(config=None, app_path=None):
        """
        Returns inputted config if valid, else a question answerer config from app's config file
        is loaded and returned.
        """
        if not config:
            return get_classifier_config("question_answering", app_path=app_path)
        return config

    @staticmethod
    def _correct_deprecated_qa_config(config):
        """
        for backwards compatibility
          if the config is supplied in deprecated format, its format is corrected and returned,
          else it is not modified and returned as-is

        deprecated usage
            >>> config = {
                    "model_type": "keyword",  # or "text", "embedder", "embedder_keyword", etc.
                    "model_settings": {
                        ...
                    }
                }

        new usage
            >>> config = {
                    "model_type": "elasticsearch",  # or "native"
                    "model_settings": {
                        "query_type": "keyword",  # or "text", "embedder", "embedder_keyword", etc.
                        ...
                    }
                }
        """
        if not config.get("model_settings", {}).get("query_type"):
            model_type = config.get("model_type")
            if not model_type:
                msg = f"Invalid 'model_type': {model_type} found while creating a QuestionAnswerer"
                raise ValueError(msg)
            if model_type in QUESTION_ANSWERER_MODEL_MAPPINGS:
                raise ValueError(
                    f"Could not find {model_type} in `model_settings` of question answerer"
                )

            msg = (
                "Using deprecated config format for Question Answerer. "
                "See https://www.mindmeld.com/docs/userguide/kb.html for more details."
            )
            warnings.warn(msg, DeprecationWarning)
            config = copy.deepcopy(config)
            model_settings = config.get("model_settings", {})
            model_settings.update({"query_type": model_type})
            config["model_settings"] = model_settings
            config["model_type"] = "elasticsearch"
        return config

    @staticmethod
    def _get_question_answerer_class(model_type: str) -> Type["BaseQuestionAnswerer"]:
        if model_type not in QUESTION_ANSWERER_MODEL_MAPPINGS:
            msg = (
                f"Did not find {model_type} in config of Question Answerer among "
                f"{[*QUESTION_ANSWERER_MODEL_MAPPINGS]}"
            )
            raise ValueError(msg)

        if model_type == "elasticsearch" and not _is_module_available("elasticsearch"):
            raise ImportError(
                "Must install the extra [elasticsearch] by running 'pip install "
                "mindmeld[elasticsearch]' to use Elasticsearch for question answering."
            )

        return QUESTION_ANSWERER_MODEL_MAPPINGS[model_type]

    @staticmethod
    def register_question_answerer_class(model_type: str, model_cls: Type["BaseQuestionAnswerer"]):
        """Registers a QA class."""
        QUESTION_ANSWERER_MODEL_MAPPINGS[model_type] = model_cls


class BaseQuestionAnswerer(ABC):
    # TODO: change name to QuestionAnswerer upon removing existing QuestionAnswerer class

    def __init__(self, app_path=None, config=None, app_namespace=None, **_kwargs):
        """
        Args:
            app_path (str, optional): The path to the directory containing the app's data. If
                provided, used to obtain default 'app_namespace' and QA configurations
            config (dict, optional): The QA config if passed directly rather than loaded from the
                app config
            app_namespace (str, optional): The namespace of the app. Used to prevent collisions
                between the indices of this app and those of other apps. If None, it's value is
                determined from the app_path.
        """
        logger.debug(f"Initializing {self.__class__.__name__}")

        if not app_path and not app_namespace:
            msg = (
                f"At least one of 'app_path' or 'app_namespace' must be inputted as arguments "
                f"while creating an instance of {self.__class__.__name__} in order to "
                f"distinctly identify the Knowledge Base indices being created."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.app_path = os.path.abspath(app_path) if app_path else app_path
        self.app_namespace = app_namespace or get_app_namespace(self.app_path)

        self._qa_config = config or get_classifier_config(
            "question_answering", app_path=self.app_path
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} query_type:{self.query_type} "
            f"app_path:{self.app_path} app_namespace:{self.app_namespace}>"
        )

    @property
    def model_type(self) -> str:
        return self._qa_config.get("model_type")

    @property
    def query_type(self) -> str:
        _query_type = self._qa_config.get("model_settings", {}).get("query_type")
        if _query_type in ALL_QUERY_TYPES:
            return _query_type
        else:
            return DEFAULT_QUERY_TYPE

    @property
    def model_settings(self) -> dict:
        model_settings = self._qa_config.get("model_settings", {})

        # add defaults
        if model_settings.get("embedder_type") == "bert":
            default_pretrained_name_or_abspath = "sentence-transformers/bert-base-nli-mean-tokens"
            pretrained_name_or_abspath = model_settings.get("pretrained_name_or_abspath")
            if not pretrained_name_or_abspath:
                msg = (
                    f"Using a default value ('{default_pretrained_name_or_abspath}') "
                    f"for the parameter 'pretrained_name_or_abspath' as NoneType value inputted."
                )
                logger.warning(msg)
                pretrained_name_or_abspath = default_pretrained_name_or_abspath
            model_settings["pretrained_name_or_abspath"] = pretrained_name_or_abspath
        elif model_settings.get("embedder_type") == "glove":
            default_token_embedding_dimension = 300
            token_embedding_dimension = model_settings.get("token_embedding_dimension")
            if not token_embedding_dimension:
                msg = (
                    f"Using a default value ('{default_token_embedding_dimension}') "
                    f"for the parameter 'token_embedding_dimension' as NoneType value inputted."
                )
                logger.warning(msg)
                token_embedding_dimension = default_token_embedding_dimension
            model_settings["token_embedding_dimension"] = token_embedding_dimension

        return model_settings

    def load_kb(self, index_name, data_file, **kwargs):
        """Loads documents from disk into the specified index in the knowledge
        base. If an index with the specified name doesn't exist, a new index
        with that name will be created in the knowledge base.

        Args:
            index_name (str): The name of the new index to be created; can be any valid string
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.

        Optional Args (used by all QA classes):
            app_namespace (str): A custom namespace of the app. Used to prevent collisions between
                the indices of two apps with same app name.
            clean (bool): Set to true if you want to delete an existing index and reindex it. If
                False (default), ElasticsearchQA just updates its index with new objects not
                deleting the old objects whereas NativeQA replaces old index with new index
                consisting of the inputted data file's KB objects.
            embedding_fields (list): List of embedding fields for the given index that can be
                directly passed-in instead of adding them to QA config or overriding QA config.
                Embedder information is generated and indexed only for the user specified fields
                and not all KB field names. If this list is empty, no fields have the embedder
                component even though 'embedder' keyword is specified in 'model_type'.

        Optional Args (Elasticsearch specific):
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
            connect_timeout (int): The amount of time for a connection to the Elasticsearch host.
        """

        if ("config" in kwargs and kwargs["config"]) or (
            "app_path" in kwargs and kwargs["app_path"]
        ):
            msg = (
                "Passing 'config' or 'app_path' to '.load_kb()' method is no longer "
                "supported. Create a Question Answerer instance with the required "
                "configurations and/or app path before calling '.load_kb()'."
            )
            raise ValueError(msg)
        self._load_kb(index_name, data_file, **kwargs)

    def get(
        self,
        index_name=None,
        size=10,
        query_type=None,
        app_namespace=None,
        **kwargs,
    ):
        """
        Args:
            index_name (str): The name of an index.
            size (int): The maximum number of records, default to 10.
            query_type (str): Whether the search is over structured, unstructured and whether to use
                              text signals for ranking, embedder signals, or both.
            id (str): The id of a particular document to retrieve.
            _sort (str): Specify the knowledge base field for custom sort.
            _sort_type (str): Specify custom sort type. Valid values are 'asc', 'desc' and
                              'distance'.
            _sort_location (dict): The origin location to be used when sorting by distance.

        Returns:
            list: A list of matching documents.
        """

        index_name = self._resolve_deprecated_index_name(index_name, kwargs.pop("index", None))
        return self._get(
            index=index_name,
            size=size,
            query_type=query_type,
            app_namespace=app_namespace,
            **kwargs,
        )

    def build_search(self, index_name=None, ranking_config=None, app_namespace=None, **kwargs):
        """Build a search object for advanced filtered search.

        Args:
            index_name (str): index name of knowledge base object.
            ranking_config (dict, optional): overriding ranking configuration parameters.
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.
        Returns:
            Search: a Search object for filtered search.
        """

        index_name = self._resolve_deprecated_index_name(index_name, kwargs.pop("index", None))
        return self._build_search(
            index=index_name,
            ranking_config=ranking_config,
            app_namespace=app_namespace,
            **kwargs,
        )

    @staticmethod
    def _resolve_deprecated_index_name(index_name, index):
        if index:
            msg = (
                "Input the index name to a question answerer method by using the argument name "
                "'index_name' instead of 'index'."
            )
            warnings.warn(msg, DeprecationWarning)
        index_name = index_name or index
        if not index_name:
            msg = "Missing one required argument: 'index_name'"
            raise TypeError(msg)
        return index_name

    @abstractmethod
    def _get(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_search(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _load_kb(self, *args, **kwargs):
        raise NotImplementedError


class QuestionAnswerer:
    """
    Backwards compatible QuestionAnswerer class

    old usages (allowed but will soon be deprecated)
        # loading KB directly through class method
        >>> QuestionAnswerer.load_kb(...)
        # instantiating a QA object from QuestionAnswerer instead of QuestionAnswererFactory
        >>> question_answerer = QuestionAnswerer(app_path, resource_loader, es_host, config)

    new usages
        >>> question_answerer = QuestionAnswererFactory.create_question_answerer(**kwargs)
        # Use the QA object's methods to load KB and get search results, instead of class methods
        >>> question_answerer.load_kb(...)
        >>> question_answerer.get(...) # .get(...) and .build_search(...)
    """

    DEPRECATION_MESSAGE = (
        "Calling QuestionAnswerer class directly will be deprecated in future versions. "
        "To instantiate a QA instance, use the QuestionAnswererFactory by calling "
        "'qa = QuestionAnswererFactory.create_question_answerer(**kwargs)'. "
        "An instantiated QA can then be used as 'qa.load_kb(...)', 'qa.get(...)', etc. "
        "See https://www.mindmeld.com/docs/userguide/kb.html for details about the various "
        "functionalities available with different question-answerers."
    )

    def __new__(
        cls,
        app_path=None,
        resource_loader=None,
        es_host=None,
        config=None,
        **kwargs,
    ):
        """
        This method is used to initialize a XxxQuestionAnswerer based on the model_type.

        To keep the code base backwards compatible, we use a '__new__()' way of creating instances
        alongside using a factory approach. For cases wherein a question-answerer is instantiated
        from 'QuestionAnswerer' class instead of 'QuestionAnswererFactory.create_question_answerer',
        this method is called before __init__ and returns an instance of a question-answerer.

        Due to this reason, see that the order of the arguments is similar to the previous version
        of QuestionAnswerer class in 'question_answerer.py'.
        """

        warnings.warn(QuestionAnswerer.DEPRECATION_MESSAGE, DeprecationWarning)

        kwargs.update(
            {
                "app_path": app_path,
                "es_host": es_host,
                "config": config,
                "resource_loader": resource_loader,
            }
        )
        return QuestionAnswererFactory.create_question_answerer(**kwargs)

    @classmethod
    def load_kb(
        cls,
        app_namespace,
        index_name,
        data_file,
        es_host=None,
        es_client=None,
        connect_timeout=2,
        clean=False,
        app_path=None,
        config=None,
        **kwargs,
    ):
        """
        Implemented to maintain backward compatibility. Should be removed in future versions.

        Args:
            app_namespace (str): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other
                apps.
            index_name (str): The name of the new index to be created.
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.
            es_host (str): The Elasticsearch host server.
            es_client (Elasticsearch): The Elasticsearch client.
            connect_timeout (int, optional): The amount of time for a
                connection to the Elasticsearch host.
            clean (bool): Set to true if you want to delete an existing index
                and reindex it
            app_path (str): The path to the directory containing the app's data
            config (dict): The QA config if passed directly rather than loaded from the app config
        """

        warnings.warn(QuestionAnswerer.DEPRECATION_MESSAGE, DeprecationWarning)

        # As a way to reduce entropy in using 'load_kb()' and it's related inconsistencies of not
        # exposing 'app_namespace' argument in '.get()' and '.build_search()', this reformatting
        # recommends that all these methods be used as instance methods and not as class methods.
        # By doing so, each QA object is meant to be used for one app_path/app_namespace and all
        # the indices in that app, while previously once could access any app's index.

        msg = (
            "Calling the 'load_kb(...)' method directly from the QuestionAnswerer object "
            "like 'QuestionAnswerer.load_kb(...)' will be deprecated. New usage: "
            "'qa = QuestionAnswererFactory.create_question_answerer(**kwargs); "
            "qa.load_kb(...)'. Note that this change might also "
            "lead to creating different QA instances for different configs. "
            "See https://www.mindmeld.com/docs/userguide/kb.html for more details. "
        )
        warnings.warn(msg, DeprecationWarning)

        # add everything except 'index_name' and 'data_file' to kwargs, and create a QA instance
        kwargs.update(
            {
                "app_namespace": app_namespace,
                "es_host": es_host,
                "es_client": es_client,
                "connect_timeout": connect_timeout,
                "clean": clean,
                "config": config,
                "app_path": app_path,
            }
        )
        question_answerer = QuestionAnswererFactory.create_question_answerer(**kwargs)

        # only retain 'connection_timeout', 'clean' information as everything else is already
        #   absorbed during instantiation above; the recommended way of passing configs to QA is
        #   by passing those details during initialization, that way there exists no discrepancies
        #   between loading and inference.
        kwargs.pop("app_namespace")
        kwargs.pop("es_host")
        kwargs.pop("es_client")
        kwargs.pop("config")
        kwargs.pop("app_path")
        question_answerer.load_kb(index_name, data_file, **kwargs)
