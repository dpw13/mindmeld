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
This module contains the question answerer component of MindMeld.
"""
import json
import logging
import numbers
import os
import pickle
import re
import unicodedata
import uuid
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians
from typing import Any, List, Union

import nltk
import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize

from ._util import _is_module_available, _get_module_or_attr as _getattr
from .question_answerer_base import (
    QuestionAnswererFactory,
    BaseQuestionAnswerer,
    DEFAULT_QUERY_TYPE,
    ALL_QUERY_TYPES,
)
from .entity_resolver import (
    EmbedderCosSimEntityResolver,
    TfIdfSparseCosSimEntityResolver,
)
from ..core import Bunch
from ..exceptions import (
    KnowledgeBaseError,
)
from ..path import (
    get_question_answerer_index_cache_file_path,
    NATIVE_QUESTION_ANSWERER_INDICES_CACHE_DEFAULT_FOLDER as DEFAULT_APP_PATH,
)
from ..query_factory import QueryFactory
from ..resource_loader import Hasher, ResourceLoader
from ..system_entity_recognizer import NoOpSystemEntityRecognizer
from ..text_preparation.text_preparation_pipeline import (
    TextPreparationPipelineFactory,
)
from ..text_preparation.tokenizers import WhiteSpaceTokenizer

# See comment in entity_resolver regarding the same pylint disable
# pylint: disable=possibly-used-before-assignment
if _is_module_available("elasticsearch"):
    from ._elasticsearch_helpers import (
        get_scoped_index_name,
    )

logger = logging.getLogger(__name__)

SORT_ORDER_ASC = "asc"
SORT_ORDER_DESC = "desc"
SORT_DISTANCE = "distance"
SORT_TYPES = {SORT_ORDER_ASC, SORT_ORDER_DESC, SORT_DISTANCE}


class NativeQuestionAnswerer(BaseQuestionAnswerer):
    """
    The question answerer is primarily an information retrieval system that provides all the
    necessary functionality for interacting with the application's knowledge base.

    This class uses Entity Resolvers in the backend to implement various underlying functionalities
    of question answerer. It consists of three important sub-classes: (1) *Indices* which maintains
    the different indices including fit entity resolvers used for inference, (2) *FieldResource*
    which forms the core of each index, encapsulating the fit resolvers and metadata related to each
    KB field, (3) *Search* class that is used to build custom search similar to what
    ElasticsearchQuestionAnswerer offers. In addition, NativeQuestionAnswerer also offers same apis
    as the Elasticsearch one- .get(), .load_kb(), .build_search().

    The created resolvers are dumped at DEFAULT_APP_PATH, whose directory serves as a common site to
    host all indices, similar to how all the indices of Elasticsearch are stored in a common
    directory on the disk.
    """

    # a common resource loaded generally used across all resolvers during loading process
    # a class var because it is also used in the underlying `Indices` class
    RESOURCE_LOADER = None

    def __init__(self, *args, **kwargs):
        """
        Args:
            resource_loader (ResourceLoader, optional): A resource loader object used by the
                underlying entity resolver models
        """
        super().__init__(*args, **kwargs)

        # update class' resource loader; use one if already passed-in
        resource_loader = kwargs.get("resource_loader")
        if not resource_loader:
            text_preparation_pipeline = (
                TextPreparationPipelineFactory.create_text_preparation_pipeline(
                    language="en", tokenizer=WhiteSpaceTokenizer()
                )
            )
            query_factory = QueryFactory.create_query_factory(
                app_path=None,
                text_preparation_pipeline=text_preparation_pipeline,
                system_entity_recognizer=NoOpSystemEntityRecognizer.get_instance(),
            )
            resource_loader = ResourceLoader.create_resource_loader(
                app_path=None, query_factory=query_factory
            )
        NativeQuestionAnswerer.RESOURCE_LOADER = resource_loader

    # pylint: disable=arguments-differ
    def _load_kb(
        self,
        index_name,
        data_file,
        app_namespace=None,
        clean=False,
        embedding_fields=None,
        **kwargs,
    ):
        """Loads documents from disk into the specified index in the knowledge
        base. If an index with the specified name doesn't exist, a new index
        with that name will be created in the knowledge base.

        Args:
            index_name (str): The name of the new index to be created.
            data_file (str): The path to the data file containing the documents
                to be imported into the knowledge base index. It could be
                either json or jsonl file.
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.
            clean (bool, optional): Set to true if you want to delete an existing index
                and reindex it
            embedding_fields (list, optional): List of embedding fields can be directly passed in
                instead of adding them to QA config
        """

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        query_type = self.query_type
        model_settings = self.model_settings

        # obtain a scoped index name using app_namespace and index_name
        scoped_index_name = get_scoped_index_name(app_namespace, index_name)

        # clean by deleting
        if clean:
            if NativeQuestionAnswerer.ALL_INDICES.is_available(scoped_index_name):
                msg = f"Index '{index_name}' exists for app '{app_namespace}', deleting index."
                logger.info(msg)
                NativeQuestionAnswerer.ALL_INDICES.delete_index(scoped_index_name)
            else:
                msg = (
                    f"Index '{index_name}' does not exist for app '{app_namespace}', "
                    f"creating a new index."
                )
                logger.warning(msg)

        # determine embedding fields
        embedding_fields = embedding_fields or model_settings.get("embedding_fields", {}).get(
            index_name, []
        )
        if embedding_fields:
            if "embedder" not in query_type:
                msg = (
                    f"Found KB fields to upload embedding (fields: {embedding_fields}) for "
                    f"index '{index_name}' but query_type configured for this QA "
                    f"({query_type}) has no 'embedder' phrase in it leading to not setting up "
                    f"an embedder model. Ignoring provided 'embedding_fields'."
                )
                logger.error(msg)
                embedding_fields = []
        else:
            if "embedder" in query_type:
                logger.warning(
                    "No embedding fields specified in the app config, continuing without "
                    "generating embeddings. "
                )

        def _doc_generator(_data_file):
            with open(_data_file) as data_fp:
                line = data_fp.readline()
                data_fp.seek(0)
                # fix related to Issue 220: https://github.com/cisco/mindmeld/issues/220
                if line.strip().startswith("["):
                    logging.debug("Loading data from a json file.")
                    docs = json.load(data_fp)
                    yield from docs
                else:
                    logging.debug("Loading data from a jsonl file.")
                    for line in data_fp:
                        doc = json.loads(line)
                        yield doc

        def match_regex(string, pattern_list):
            return any(re.match(pattern, string) for pattern in pattern_list)

        all_id2value = {}  # a mapping from id to value(s) for each kb field
        all_ids = {}  # maintained to keep a record of order-preserved-doc-ids of the knowledge base
        for doc in _doc_generator(data_file):
            # determine _id; once determined, it remains same for all keys of the doc
            _id = doc.get("id")
            if _id in all_ids:
                msg = f"Found a duplicate id {_id} while processing {data_file}. "
                _id = uuid.uuid4()
                msg += f"Replacing it and assigning a new id: {_id}"
                logger.warning(msg)
            if not _id:
                _id = uuid.uuid4()
                msg = (
                    f"Found an entry in {data_file} without a corresponding id. "
                    f"Assigning a randomly generated new id ({_id}) for this KB object."
                )
                logger.warning(msg)
            _id = str(_id)
            all_ids.update({_id: None})
            # update data
            for key, value in doc.items():
                if key not in all_id2value:
                    all_id2value[key] = {}
                all_id2value[key].update({_id: value})

        if clean:
            NativeQuestionAnswerer.ALL_INDICES.delete_index(scoped_index_name)

        index_resources = {}
        # Fetch an index if it already exists
        try:
            index_resources = NativeQuestionAnswerer.ALL_INDICES.get_index(scoped_index_name)
        except KnowledgeBaseError:
            pass
        for kb_field_name, id2value in all_id2value.items():
            field_resource = index_resources.get(kb_field_name)
            if not field_resource:
                field_resource = NativeQuestionAnswerer.FieldResource(
                    index_name=scoped_index_name, field_name=kb_field_name
                )
            field_resource.update_resource(
                id2value,
                has_text_resolver=("text" in query_type or "keyword" in query_type),
                has_embedding_resolver=match_regex(kb_field_name, embedding_fields),
                resolver_model_settings=model_settings,  # same settings for all fields
                clean=clean,
                processor_type="text" if "text" in query_type else "keyword",
                resource_loader=NativeQuestionAnswerer.RESOURCE_LOADER,  # one to all resolvers
            )
            index_resources.update({kb_field_name: field_resource})

        # update and dump
        NativeQuestionAnswerer.ALL_INDICES.update_index_and_persist(
            scoped_index_name, index_resources, [*all_ids.keys()]
        )

    def _get(self, index, size=10, query_type=None, app_namespace=None, **kwargs):
        doc_id = kwargs.get("id")

        query_type = query_type or self.query_type

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        # If an id was passed in, simply retrieve the specified document
        if doc_id:
            logger.info("Retrieve object from KB: index= '%s', id= '%s'.", index, doc_id)
            s = self.build_search(index, app_namespace=app_namespace)

            if NativeQuestionAnswerer.FieldResource.is_number(doc_id):
                # if a number, look for that exact doc_id; no fuzzy match like in Elasticsearch QA
                s = s.filter(query_type=query_type, field="id", et=doc_id)  # et == equals to
            else:
                s = s.filter(query_type=query_type, id=doc_id)
                # TODO: should consider s.query() to do fuzzy match like in Elasticsearch?
                # s = s.query(query_type=query_type, id=doc_id)
            results = s.execute(size=size)
            return results

        field = kwargs.pop("_sort", None)
        sort_type = kwargs.pop("_sort_type", None)
        location = kwargs.pop("_sort_location", None)

        s = self.build_search(index, app_namespace=app_namespace).query(
            query_type=query_type, **kwargs
        )
        if field and (sort_type or location):
            s.sort(field, sort_type=sort_type, location=location)

        results = s.execute(size=size)
        return results

    def _build_search(self, index, ranking_config=None, app_namespace=None):
        """Build a search object for advanced filtered search.

        Args:
            index (str): index name of knowledge base object.
            ranking_config (dict, optional): overriding ranking configuration parameters.
            app_namespace (str, optional): The namespace of the app. Used to prevent
                collisions between the indices of this app and those of other apps.
        Returns:
            Search: a Search object for filtered search.
        """

        if ranking_config:
            msg = f"'ranking_config' is currently discarded in {self.__class__.__name__}."
            logger.warning(msg)
            ranking_config = None

        # fix related to Issue 219: https://github.com/cisco/mindmeld/issues/219
        app_namespace = app_namespace or self.app_namespace

        # get index name with app scope, and make it ready for inference
        scoped_index_name = get_scoped_index_name(app_namespace, index)
        return NativeQuestionAnswerer.Search(index=scoped_index_name)

    class Indices:
        """An object that hold all the indices for an app_path

        'self._indices' has the following dictionary format, with keys as the index name and
        the value as the metadata of that index

        '''
            {
                index_name1: {key11: FieldResource11, key12: FieldResource12, ...},
                index_name2: {key21: FieldResource21, key22: FieldResource22, ...},
                index_name3: {...},
                ...
            }
        '''

        Index metadata includes metadata of each field found in the KB. The metadata for each field
        is encapsulated in a `FieldResource` object which in-turn constitutes of metadata related
        to that specific field in the KB (across all ids in the KB) along with information such as
        what data-type that field belongs to (number, date, etc.), field name, & hash of the stored
        data. See `FieldResource` class docstrings for more details.
        """

        def __init__(self, app_path):
            """
            Args:
                app_path (str): A folder wherein the indices are stored
            """
            self.app_path = app_path
            self._indices = {}  # Dict[str, Dict[str, FieldResource]]
            # maintain a record of all doc ids of the indices when creating it from KB data
            self._indices_all_ids = {}  # Dict[str, List[str]]

        def __contains__(self, index_name):
            if (index_name not in self._indices) ^ (index_name not in self._indices_all_ids):
                msg = (
                    f"Found an index name ({index_name}) present in only one of "
                    f"`self._indices` and `self._indices_all_ids`. Maybe an error during "
                    f"updating or loading the index?"
                )
                logger.debug(msg)
                raise KeyError(msg)
            return index_name in self._indices

        def _get_index_cache_path(self, index_name, app_path=None):
            """
            Returns a  ache path for a specified index name using a (predetermined) formatted path.
            If an app_path is specified, it is substituted for a default app path in that formatted
            string.

            Args:
                index_name (str): A scoped index name
                app_path (str, optional): The Mindmeld application's path where the index needs to
                    be located. If unspecified, a default path is used.

            Returns:
                str: a path to cache indices

            Example:
                returns: ~/.cache/mindmeld/question_answerers/food_ordering$restaurants.pkl
                when:    app_path=~/.cache/mindmeld/question_answerers and
                when:    index_name = food_ordering$restaurants
            """
            app_path = app_path or self.app_path
            return get_question_answerer_index_cache_file_path(app_path, index_name)

        # 'get' methods for accessing indices that are already loaded into memory

        def get_all_ids(self, index_name):
            """
            Returns all ids observed in the KB for the specified index name in chronological order.
            The specified index must already be loaded into the memory to obtain ids.

            Args:
                index_name (str): A scoped index name

            Returns:
                List[str]: a list of ids observed in the KB during load-kb

            Raises:
                KeyError: When the specified (scoped) index name is not found in memory
            """
            try:
                return self._indices_all_ids[index_name]
            except KeyError as e:
                msg = (
                    f"Index {index_name} does not exist in scope of {self.app_path}. "
                    f"Consider creating or loading it before calling '.get_all_ids()'."
                )
                raise KeyError(msg) from e

        def get_metadata(self, index_name):
            """
            Returns index's FieldResources' metadata objects if the index is available in memory.

            Args:
                index_name (str): A scoped index name

            Returns:
                metadata (Dict[str, Bunch]): metadata associated with each field name of the index

            Raises:
                KeyError: When the specified (scoped) index name is not found in memory
            """
            try:
                metadata = {
                    field_name: field_resource.to_metadata()
                    for field_name, field_resource in self._indices[index_name].items()
                }
            except KeyError as e:
                msg = (
                    f"Index {index_name} does not exist in scope of {self.app_path}. "
                    f"Consider creating or loading it before calling '.get_all_ids()'."
                )
                raise KeyError(msg) from e

            return metadata

        def is_available(self, index_name):
            """
            Checks for availability of a specified index name both in memory (i.e. if already
            loaded into memory at the time of checking) as well as in the cache path where are all
            the indices' metadata are stored.

            Args:
                index_name (str): A scoped index name for checking its availability

            Returns:
                bool: True if index name is available in memory or in cache directory, else False
            """
            return index_name in self or os.path.exists(self._get_index_cache_path(index_name))

        def get_index(self, index_name):
            """
            Returns the index corresponding to the specified index name. If the index is not found
            in memory, this method looks up the cache path to obtain the index.

            Args:
                index_name (str): A scoped index name

            Returns:
                index_resources (Dict[str, FieldResource]): a dictionary of FieldResource, one for
                    each field name in the index

            Raises:
                KnowledgeBaseError: if the specified index name is unavailable both in memory as
                    well as disk
            """
            if self.is_available(index_name):
                if index_name not in self:
                    msg = f"Loading metadata for the scoped index name from disk: {index_name}"
                    logger.info(msg)
                    cache_path = self._get_index_cache_path(index_name)
                    with open(cache_path, "rb") as opfile:
                        metadata_objects = pickle.load(opfile)
                    index_all_ids = metadata_objects.pop("__all_ids")
                    index_resources = {}
                    for field_name, cache_object in metadata_objects.items():
                        field_resource = NativeQuestionAnswerer.FieldResource.from_metadata(
                            cache_object
                        )
                        field_resource.load_resolvers(
                            resource_loader=NativeQuestionAnswerer.RESOURCE_LOADER
                        )
                        index_resources.update({field_name: field_resource})
                    self._indices.update({index_name: index_resources})
                    self._indices_all_ids.update({index_name: index_all_ids})
            else:
                msg = (
                    f"Consider creating indices first before using them. Specified scoped "
                    f"index name {index_name} not found in list of known indices."
                )
                raise KnowledgeBaseError(msg)

            return self._indices[index_name]

        def delete_index(self, index_name):
            """
            Deletes the index both from memory as well as disk
            """
            if index_name in self:
                del self._indices[index_name]  # free the pointer

            # clear index dump cache path, if required
            # Note: This might not delete the entity resolver data and need methods for doing same.
            cache_path = self._get_index_cache_path(index_name)
            if cache_path and os.path.exists(cache_path):
                os.remove(cache_path)

        def update_index_and_persist(self, index_name, index_resources, index_all_ids):
            """
            Updates the specified index's resources in the memory as well as dumps the metadata into
            disk for fast loading time later on. Note that this method is best used with the
            a load_kb() method wherein fit resources are frirst created before updating indices.

            During reloading of an index, use self.get_index_metadata() as well as
            fieldResource.update_resource() to obtain back a fit index.

            Args:
                index_name (str): A scoped index name for loading metadata
                index_resources (Dict[str, FieldResource]): Dict curated with all the KB fields and
                    their FieldResources
                 index_all_ids (List[str]): List of all ids observed for this index in the order
                    they are present in the KB.
            """

            # update memory
            self._indices.update({index_name: index_resources})
            self._indices_all_ids.update({index_name: index_all_ids})

            # dump to disk
            cache_path = self._get_index_cache_path(index_name)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            metadata = self.get_metadata(index_name)
            metadata.update({"__all_ids": index_all_ids})
            with open(cache_path, "wb") as opfile:
                pickle.dump(metadata, opfile)

    class FieldResourceDataHelper:
        """A class that holds methods to aid validating, formatting and scoring different data
        types in a FieldResource object.
        """

        DATA_TYPES = ["bool", "number", "string", "date", "location", "unknown"]

        DATE_FORMATS = (
            "%Y",
            "%d %b",
            "%d %B",
            "%b %d",
            "%B %d",
            "%b %Y",
            "%B %Y",
            "%b, %Y",
            "%B, %Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%b %d %Y",
            "%B %d %Y",
            "%b %d,%Y",
            "%B %d,%Y",
            "%d %b, %Y",
            "%d %B, %Y",
            "%d %b %Y",
            "%d %B %Y",
            "%d %b,%Y",
            "%d %B,%Y",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%d/%m/%Y",
            "%d/%m/%y",
        )

        @property
        def data_types(self):
            return self.__class__.DATA_TYPES

        @property
        def date_formats(self):
            return self.__class__.DATE_FORMATS

        @staticmethod
        def _get_date(value: str):
            value_date = None

            if not isinstance(value, str):
                return value_date

            # check if value can be resolved as-is
            for fmt in NativeQuestionAnswerer.FieldResourceDataHelper.date_formats:
                try:
                    value_date = datetime.strptime(value, fmt)
                    return value_date
                except (ValueError, TypeError):
                    pass

            # check if value can be resolved with some modifications
            for fmt in NativeQuestionAnswerer.FieldResourceDataHelper.date_formats:
                for val in [value.replace("/", " "), value.replace("-", " ")]:
                    if val != value:
                        try:
                            value_date = datetime.strptime(val, fmt)
                            return value_date
                        except (ValueError, TypeError):
                            pass

            return value_date

        @staticmethod
        def _get_location(value: Union[dict, str, List]):
            # convert it into standard format, e.g. "37.77,122.41"

            if isinstance(value, dict) and "lat" in value and "lon" in value:
                # eg. {"lat": 37.77, "lon": 122.41}
                return ",".join([str(value["lat"]), str(value["lon"])])
            elif isinstance(value, str) and "," in value and len(value.split(",")) == 2:
                # eg. "37.77,122.41"
                return value.strip()
            elif (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], numbers.Number)
                and isinstance(value[1], numbers.Number)
            ):
                # eg. [37.77, 122.41]
                return ",".join([str(_value) for _value in value])

            return None

        @staticmethod
        def is_bool(value):
            return isinstance(value, bool)

        @staticmethod
        def is_number(value):
            return isinstance(value, numbers.Number)

        @staticmethod
        def is_string(value):
            return isinstance(value, str)

        @staticmethod
        def is_list_of_strings(value):
            return (
                isinstance(value, (list, set))
                and len(value) > 0
                and all(isinstance(val, str) for val in value)
            )

        @staticmethod
        def is_date(value):
            if NativeQuestionAnswerer.FieldResourceDataHelper._get_date(value):
                return True
            return False

        @staticmethod
        def is_location(value):
            if NativeQuestionAnswerer.FieldResourceDataHelper._get_location(value):
                return True
            return False

        @staticmethod
        def number_scorer(some_number):
            return some_number

        @staticmethod
        def date_scorer(value):
            """
            ascertains a suitable date format for input and
            returns number of days from origin date as score
            """
            origin_date = datetime.now()
            target_date = NativeQuestionAnswerer.FieldResourceDataHelper._get_date(value)
            if not target_date:
                target_date = origin_date
            # TODO: should this be configurable to days or seconds based on the app?
            return (target_date - origin_date).days

        @staticmethod
        def location_scorer(some_location, source_location):
            """
            Uses Haversine formula to find distance between two coordinates
            references: https://en.wikipedia.org/wiki/Haversine_formula and
                        http://www.movable-type.co.uk/scripts/latlong.html

            Args:
                some_location (str): latitude and longitude supplied as
                    comma separated strings, eg. "37.77,122.41"
                source_location (str): latitude and longitude supplied as
                    comma separated strings, eg. "37.77,122.41"

            Returns:
                float: distance between the coordinates in kilometers

            Example 1:
                >>> point_1 = "37.78953146306901,-122.41160227491551" # SF in CA
                >>> point_2 = "47.65182346406002, -122.36765696283909" # Seattle in WA
                >>> location_scorer(point_1, point_2)
                >>> # 1096.98 kms (approx. points on Google maps says 680.23 miles/1094.72 kms)
            Example 2:
                >>> point_3, point_4 = "52.2296756,21.0122287", "52.406374,16.9251681"
                >>> location_scorer(point_3, point_4)
                >>> # 278.54 kms
            """

            some_location = NativeQuestionAnswerer.FieldResourceDataHelper._get_location(
                some_location
            )
            source_location = NativeQuestionAnswerer.FieldResourceDataHelper._get_location(
                source_location
            )

            # pylint: disable=invalid-name
            R = 6373.0  # constant based on Haversine formula
            lat1, lon1 = [radians(float(ii.strip())) for ii in some_location.split(",")]
            lat2, lon2 = [radians(float(ii.strip())) for ii in source_location.split(",")]
            dlon, dlat = lon2 - lon1, lat2 - lat1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance_haversine_formula = R * c

            return distance_haversine_formula

        @staticmethod
        def min_max_normalizer(list_of_numbers):
            _min = min(list_of_numbers)
            list_of_numbers = [num - _min for num in list_of_numbers]
            _max = max(list_of_numbers)
            _max = 1.0 if not _max else _max
            list_of_numbers = [num / _max for num in list_of_numbers]
            return list_of_numbers

    class FieldResourceHelper(FieldResourceDataHelper):
        @staticmethod
        def _auto_string_processor(string_or_strings, query_type, language="english"):
            if language != "english":
                # TODO: implement support for non-english texts
                msg = "Only allowed language for text processing in QA is 'english'. "
                raise NotImplementedError(msg)

            try:
                english_stop_words = set(nltk_stopwords.words(language))
            except LookupError:
                nltk.download("stopwords")
                english_stop_words = set(nltk_stopwords.words(language))

            english_stemmer = PorterStemmer()

            try:
                nltk_word_tokenize(" ")
            except LookupError:
                nltk.download("punkt")

            def lowercase(string):
                return string.lower()

            def strip_accents(string):
                return "".join(
                    (
                        c
                        for c in unicodedata.normalize("NFD", string)
                        if unicodedata.category(c) != "Mn"
                    )
                )

            def keyword_processor(string):
                # TODO: can add char_filters like Elasticsearch; see 'keyword_match_analyzer' in
                #       'components/_elasticsearch_hepers.py'
                return strip_accents(lowercase(str(string)))

            def text_processor(string):
                string = strip_accents(lowercase(str(string)))
                word_tokens = nltk_word_tokenize(string)
                filtered_string = " ".join(
                    [english_stemmer.stem(w) for w in word_tokens if w not in english_stop_words]
                )
                return filtered_string

            if "keyword" in query_type:
                processor = keyword_processor
            elif "text" in query_type:
                processor = text_processor
            else:
                raise ValueError("query_type in processor must contain 'text' or 'keyword' string")

            if isinstance(string_or_strings, str):
                return processor(string_or_strings)
            elif isinstance(string_or_strings, (list, set)):
                return [processor(string) for string in string_or_strings]
            else:
                raise ValueError("Input to auto processor must be string or list of strings")

        def _resolve_data_type(self, value: Any) -> str:
            try:
                value = value.strip()
            except AttributeError:
                pass

            if self.is_location(value):
                observed_data_type = "location"
            elif self.is_bool(value):
                observed_data_type = "bool"
            elif self.is_number(value):
                observed_data_type = "number"
            elif self.is_string(value) or self.is_list_of_strings(value):
                observed_data_type = "string"
            elif self.is_date(value):
                observed_data_type = "date"
            else:
                observed_data_type = "unknown"

            return observed_data_type

        def _validate_and_reformat_value(
            self,
            known_data_type,
            value,
            _id=None,
            field_name=None,
            index_name=None,
        ):
            def _raise_error():
                errmsg = (
                    f"Formatting error for the field {field_name}"
                    f"{' in doc id' if _id else ''} {str(_id) if _id else ''} "
                    f"in index {index_name}. Found an unexpected type {type(value)} "
                    f"but expected the field value to have type {known_data_type}"
                )
                logger.error(errmsg)
                raise TypeError(errmsg)

            if known_data_type == "location":
                if not self.is_location(value):
                    _raise_error()
            elif known_data_type == "bool":
                if not self.is_bool(value):
                    _raise_error()
            elif known_data_type == "number":
                if not self.is_number(value):
                    _raise_error()
            elif known_data_type == "string":
                if self.is_string(value):
                    value = value.strip()
                elif self.is_list_of_strings(value):
                    value = [val.strip() for val in value]
                else:
                    _raise_error()
            elif known_data_type == "date":
                value = value.strip()
                if not self.is_date(value):
                    _raise_error()

            return value

        def _get_resolvers_entity_map(self, id2value):
            """
            converts id2value into an entity map format and returns it
            """

            def _get_resolvers_whitelist(value):
                if isinstance(value, (set, list)):
                    return list(value)[1:]
                return []

            # new: https://github.com/cisco/mindmeld/issues/291
            #   making 'value' into a list for cname and whitelist conversion.
            #   If more than one items in 'value', all items after first one go into whitelist
            entity_map = {
                "entities": [
                    {
                        "id": _id,
                        "cname": self.get_resolvers_cname(value),
                        "whitelist": _get_resolvers_whitelist(value),
                    }
                    for _id, value in id2value.items()
                ]
            }
            return entity_map

        @staticmethod
        def get_resolvers_cname(value):
            if isinstance(value, (set, list)):
                return list(value)[0]
            return value

    class FieldResource(FieldResourceHelper):
        """
        An object encapsulating all resources necessary for search/filter/sort-ing on any field in
        the Knowledge Base. This class should only be used as part of `Indices` class and not in
        isolation.

        This class currently supports:
        - location strings,
        - date strings,
        - boolean,
        - number,
        - strings, and
        - list of strings.

        Any other data type (eg. dictionary type) is currently not supported and is marked as an
        'unknown' data type. Such unknown data types fields do not have any associated resolvers.
        """

        def __init__(self, index_name, field_name):
            # details to establish a scoped field name
            self.index_name = index_name
            self.field_name = field_name

            # vars that contain data of the field
            self.data_type = None
            self.id2value = {}  # Resolvers don't dump data; so no data duplication on the disk
            self.hash = None  # to identify any data changes before build resolver

            # details to create any required resolvers
            self.processor_type = None
            self.has_text_resolver = None
            self.has_embedding_resolver = None

            # required resolvers
            self._text_resolver = None  # an entity resolver if string type data
            self._embedding_resolver = None  # an embedding based entity resolver

        def __repr__(self):
            return (
                f"{self.__class__.__name__} "
                f"field_name: {self.field_name} "
                f"data_type: {self.data_type} "
                f"has_text_resolver: {self.has_text_resolver} "
                f"has_embedding_resolver: {self.has_embedding_resolver}"
            )

        def update_resource(
            self,
            id2value,
            has_text_resolver,
            has_embedding_resolver,
            resolver_model_settings=None,
            clean=False,
            app_path=DEFAULT_APP_PATH,
            processor_type="keyword",
            resource_loader=None,
        ):
            """
            Updates a field resource by fitting with latest data (if id2value is passed) or by
            loading already fit resolvers if no data changes take place.

            While loading from metadata, the `self.id2value` data is also loaded, which is in turn
            used to create the `new_hash`. Even if the `new_hash` is same as `self.hash` (implying
            no data changes), if the `self._text_resolver` is not fit but is required, the resolver
            is loaded instead of fitting.

            Args:
                id2value (dict): a mapping between documnet ids & values of the chosen KB field
                has_text_resolver (bool): If a tfidf resolver is to be created
                has_embedding_resolver (bool): If a embedder resolver is to be created
                resolver_model_settings (dict): A dictionary cocnsisting of model settings for the
                    resolver models. Currently, the same setting is passed to all kinds of resolver
                    models.
                clean (bool, optional): if True, resolvers are fit with clean=True
                app_path (str, optional): a path to create cache for embedder resolver
                processor_type (str, optional, "text" or "keyword"): processor for tfidf resolver
                resource_loader (ResourceLoader, optional): a resource loader object
            """

            # reformat id2value's values if required and obtain data type from the data
            for _id, value in id2value.items():
                # ignore null values
                if not isinstance(value, bool) and not value:
                    continue
                # first non empty value will determine the data type of this field if not already
                #   determined. will be 'unknown' if all values are empty or if there is a ambiguity
                #   in deciding the data type.
                if self.data_type is None:
                    self.data_type = self._resolve_data_type(value)
                # validation and re-formatting to update database, no change for unknown data type
                try:
                    value = self._validate_and_reformat_value(
                        self.data_type,
                        value,
                        _id,
                        self.field_name,
                        self.index_name,
                    )
                except TypeError:
                    # implies that this field had different observed data type across different docs
                    self.data_type = "unknown"
                self.id2value.update({_id: value})
            # self.id2value is None (i.e initialized default) in cases wherein empty id2value is
            # passed in input arguments or if all KB objects have null data for this field
            if not self.id2value:
                msg = f"Found no data for field {self.field_name}. "
                logger.warning(msg)
                self.data_type = "unknown"

            # compute hash on latest data and ascertain resolver(s) requirement
            new_hash = None
            if self.data_type in ["bool", "number", "location", "unknown"]:
                # discard input arguments as resolvers are not applicable to these data types
                if has_text_resolver or has_embedding_resolver:
                    msg = (
                        f"Unable to create any resolver for the field {self.field_name} due to "
                        f"its marked data type '{self.data_type}'. "
                    )
                    logger.info(msg)
                self.has_text_resolver = False
                self.has_embedding_resolver = False
                return
            else:  # ["string", "date"]
                self.has_text_resolver = has_text_resolver
                self.has_embedding_resolver = has_embedding_resolver
                if not self.has_text_resolver and not self.has_embedding_resolver:
                    msg = (
                        f"At least one of text or embedder resolver can be applied "
                        f"for string type data field ({self.field_name}) but continuing "
                        f"without fitting any resolvers due to your input 'query_type'"
                        f"configuration. This might limit your search space during inference!"
                    )
                    logger.warning(msg)
                    return
                new_hash = Hasher(algorithm="sha256").hash(
                    string=json.dumps(self.id2value, sort_keys=True)
                )

            # tfidf based text resolver
            if self.has_text_resolver and (
                (new_hash != self.hash) or (self.processor_type != processor_type)
            ):
                msg = (
                    f"Creating a text resolver for field '{self.field_name}' in "
                    f"index '{self.index_name}'."
                )
                logger.info(msg)
                # update processor type
                if processor_type not in ["text", "keyword"]:
                    msg = (
                        f"Expected 'processor_type' to be among ['text', "
                        f"'keyword'] but found to be of value '{processor_type}'"
                    )
                    raise ValueError(msg)
                self.processor_type = processor_type
                # obtain a cache path
                resolver_cache_path = get_question_answerer_index_cache_file_path(
                    app_path,
                    get_scoped_index_name(
                        get_scoped_index_name(self.index_name, self.field_name),
                        "text_resolver",
                    ),
                )
                # create a new resolver and fit
                resolver_model_settings = resolver_model_settings or {}
                self._text_resolver = TfIdfSparseCosSimEntityResolver(
                    app_path=app_path,
                    entity_type=get_scoped_index_name(self.index_name, self.field_name),
                    config={
                        "model_settings": {
                            **resolver_model_settings,
                            "augment_max_synonyms_embeddings": False,
                        }
                    },
                    resource_loader=resource_loader,
                )
                entity_map = self._get_resolvers_entity_map(
                    dict(
                        zip(
                            self.id2value.keys(),
                            self._auto_string_processor(
                                [*self.id2value.values()], self.processor_type
                            ),
                        )
                    )
                )
                self._text_resolver.fit(entity_map=entity_map, clean=clean)
                # dump
                # ex: ~/.cache/mindmeld/question_answerers/
                #           {food_ordering}${restaurants}${field_name}
                #               .pkl
                #               .pkl.hash
                #               .config.pkl
                self._text_resolver.dump(resolver_cache_path)

            # embedder resolver
            if self.has_embedding_resolver and new_hash != self.hash:
                msg = (
                    f"Creating an embedder resolver for field '{self.field_name}' in "
                    f"index '{self.index_name}'."
                )
                logger.info(msg)
                # obtain a cache path
                resolver_cache_path = get_question_answerer_index_cache_file_path(
                    app_path,
                    get_scoped_index_name(
                        get_scoped_index_name(self.index_name, self.field_name),
                        "embedder_resolver",
                    ),
                )
                # create a new resolver and fit
                resolver_model_settings = resolver_model_settings or {}
                self._embedding_resolver = EmbedderCosSimEntityResolver(
                    app_path=app_path,
                    entity_type=get_scoped_index_name(self.index_name, self.field_name),
                    config={"model_settings": {**resolver_model_settings}},
                    resource_loader=resource_loader,
                )
                entity_map = self._get_resolvers_entity_map(
                    self.id2value
                )  # use same data as text resolver but without any processing!
                self._embedding_resolver.fit(entity_map=entity_map, clean=clean)
                # dump
                # ex: ~/.cache/mindmeld/question_answerers/
                #           {food_ordering}${restaurants}${field_name}
                #               .pkl.hash
                #               .config.pkl
                #               .embedder_cache.pkl
                self._embedding_resolver.dump(resolver_cache_path)

            self.hash = new_hash

        def load_resolvers(self, app_path=DEFAULT_APP_PATH, resource_loader=None):
            """
            Loads a field resource by fitting with latest data (if id2value is passed) or by
            loading already fit resolvers if no data changes take place.

            Args:
                app_path (str, optional): a path to create cache for embedder resolver
                resource_loader (ResourceLoader, optional): a resource loader object
            """

            def _err_fn(msg):
                logger.error(msg)
                raise ValueError(msg)

            err_msg = (
                "Error in loading FieldResource: "
                "'{name}' cannot be '{value}' when loading a "
                "FieldResource from metadata object. Your metadata object might have "
                "been corrupted.Consider loading kb with 'clean=True'."
            )
            if self.data_type is None:
                _err_fn(err_msg.format(name="data_type", value=self.data_type))
            if self.has_text_resolver is None:
                _err_fn(err_msg.format(name="has_text_resolver", value=self.has_text_resolver))
            if self.has_embedding_resolver is None:
                _err_fn(
                    err_msg.format(
                        name="has_embedding_resolver",
                        value=self.has_embedding_resolver,
                    )
                )

            if self.data_type in ["bool", "number", "location", "unknown"]:
                # discard input arguments as resolvers are not applicable to these data types
                if self.has_text_resolver or self.has_embedding_resolver:
                    msg = (
                        f"Unable to create any resolver for the field {self.field_name} due to "
                        f"its marked data type '{self.data_type}'. "
                    )
                    logger.info(msg)
                self.has_text_resolver = False
                self.has_embedding_resolver = False
                return
            else:  # ["string", "date"]
                if not self.has_text_resolver and not self.has_embedding_resolver:
                    msg = (
                        f"At least one of text or embedder resolver can be applied "
                        f"for string type data field ({self.field_name}) but continuing "
                        f"without fitting any resolvers due to your input 'query_type'"
                        f"configuration. This might limit your search space during inference!"
                    )
                    logger.warning(msg)
                    return

            # tfidf based text resolver
            if self.has_text_resolver:
                msg = (
                    f"Loading a text resolver for field '{self.field_name}' in "
                    f"index '{self.index_name}'."
                )
                logger.info(msg)
                if self.processor_type is None:
                    _err_fn(err_msg.format(name="processor_type", value=self.processor_type))
                try:
                    # obtain a cache path
                    resolver_cache_path = get_question_answerer_index_cache_file_path(
                        app_path,
                        get_scoped_index_name(
                            get_scoped_index_name(self.index_name, self.field_name),
                            "text_resolver",
                        ),
                    )
                    # create a new instance of resolver and load it
                    self._text_resolver = TfIdfSparseCosSimEntityResolver(
                        app_path=app_path,
                        entity_type=get_scoped_index_name(self.index_name, self.field_name),
                        resource_loader=resource_loader,
                    )
                    entity_map = self._get_resolvers_entity_map(
                        dict(
                            zip(
                                self.id2value.keys(),
                                self._auto_string_processor(
                                    [*self.id2value.values()],
                                    self.processor_type,
                                ),
                            )
                        )
                    )
                    self._text_resolver.load(path=resolver_cache_path, entity_map=entity_map)
                except Exception as e:
                    msg = (
                        "Couldn't load a text resolver from cache path. Consider "
                        "calling the 'load_kb()' method with argument 'clean=True'."
                    )
                    logger.error(msg)
                    raise KnowledgeBaseError(msg) from e

            # embedder resolver
            if self.has_embedding_resolver:
                msg = (
                    f"Loading an embedder resolver for field '{self.field_name}' in "
                    f"index '{self.index_name}'."
                )
                logger.info(msg)
                try:
                    # obtain a cache path
                    resolver_cache_path = get_question_answerer_index_cache_file_path(
                        app_path,
                        get_scoped_index_name(
                            get_scoped_index_name(self.index_name, self.field_name),
                            "embedder_resolver",
                        ),
                    )
                    # create a new instance of resolver and load it
                    self._embedding_resolver = EmbedderCosSimEntityResolver(
                        app_path=app_path,
                        entity_type=get_scoped_index_name(self.index_name, self.field_name),
                        resource_loader=resource_loader,
                    )
                    entity_map = self._get_resolvers_entity_map(
                        self.id2value
                    )  # use same data as text resolver but without any processing!
                    self._embedding_resolver.load(path=resolver_cache_path, entity_map=entity_map)
                except Exception as e:
                    msg = (
                        "Couldn't load embedder resolver from cache path. Consider "
                        "calling the 'load_kb()' method with argument 'clean=True'."
                    )
                    logger.error(e)
                    raise KnowledgeBaseError(msg) from e

        def do_search(self, query_type, value, allowed_ids=None):
            """
            Retrieves doc ids with corresponding similarity scores for the given query

            Args:
                query_type (str): one of ALL_QUERY_TYPES
                value (str): A string to do similarity search
                allowed_ids (iterable, optional): if not None, only docs containing these ids are
                    populated in the results
            Returns:
                dict: a mapping between _ids recorded in this field and their corresponding scores
            """

            self._do_search_validation(query_type)

            value = self._validate_and_reformat_value(self.data_type, value)

            def update_scores(
                resolver,
                value_string,
                this_field_scores,
                n_scores,
                allowed_cnames,
            ):
                # obtain synonyms' scores without sorting! (saves compute time)
                # also, do matching against selected cnames only
                predictions = resolver.predict(
                    value_string, top_n=None, allowed_cnames=allowed_cnames
                )
                # retain only top scored entries for each id
                _best_scores = {}
                for prediction in predictions:
                    _id, _score = prediction["id"], prediction["score"]
                    if _id not in _best_scores:
                        _best_scores[_id] = _score
                    else:
                        _best_scores[_id] = max(_best_scores[_id], _score)
                # for missing doc ids, populate the minimum score
                # in cases where there was no training data for resolver, all ids are absent
                #   in the returned predictions! And then the _best_scores will be empty
                if _best_scores:
                    min_best_scores = min(*_best_scores.values())
                    for _id in self.id2value.keys():
                        if allowed_ids and _id not in allowed_ids:
                            continue
                        if _id not in _best_scores:
                            _best_scores[_id] = min_best_scores
                    # retain only top scored entries for each id
                    this_field_scores = {
                        _id: (this_field_scores.get(_id, 0.0) + _score) / (n_scores + 1)
                        for _id, _score in _best_scores.items()
                    }
                    n_scores += 1

                # if no predictions are obtained from resolver, these values are same as input
                return this_field_scores, n_scores

            # maps doc ids with their similarity scores
            this_field_scores = {}
            n_scores = 0

            if ("text" in query_type or "keyword" in query_type) and self._text_resolver:
                # get processor type, process the value and then obtain similarities
                processor_type = "text" if "text" in query_type else "keyword"
                if processor_type != self.processor_type:
                    msg = (
                        f"Using different text processing during loading KB "
                        f"({self.processor_type}) vs during inference ({processor_type}) "
                        f"for the field {self.field_name} in index {self.index_name}"
                    )
                    logger.warning(msg)
                new_value = self._auto_string_processor(value, processor_type)
                if allowed_ids:
                    filtered_cnames = [
                        self.get_resolvers_cname(self._auto_string_processor(val, processor_type))
                        for _id, val in self.id2value.items()
                        if _id in allowed_ids
                    ]
                else:
                    filtered_cnames = None
                this_field_scores, n_scores = update_scores(
                    self._text_resolver,
                    new_value,
                    this_field_scores,
                    n_scores,
                    filtered_cnames,
                )
            elif "text" in query_type or "keyword" in query_type:
                msg = (
                    f"No text based resolver configured for field {self.field_name} "
                    f"in index {self.index_name}."
                )
                logger.warning(msg)

            if "embedder" in query_type and self._embedding_resolver:
                if allowed_ids:
                    filtered_cnames = [
                        self.get_resolvers_cname(val)
                        for _id, val in self.id2value.items()
                        if _id in allowed_ids
                    ]
                else:
                    filtered_cnames = None
                this_field_scores, n_scores = update_scores(
                    self._embedding_resolver,
                    value,
                    this_field_scores,
                    n_scores,
                    filtered_cnames,
                )
            elif "embedder" in query_type:
                msg = (
                    f"No embedder based resolver configured for field {self.field_name} "
                    f"in index {self.index_name}."
                )
                logger.warning(msg)

            # Case where-in no resolver exists (eg. "unknown" data type)
            if not this_field_scores:
                this_field_scores = {_id: 0.0 for _id in self.id2value.keys()}

            return this_field_scores

        def do_filter(
            self,
            allowed_ids,
            filter_text=None,
            gt=None,
            gte=None,
            lt=None,
            lte=None,
            et=None,
            boolean=None,
        ):
            """
            Filters a list of docs to a subset based on some criteria such as a boolean value
            or {>,<,=} operations or a text snippet.
            """

            if not allowed_ids:
                return allowed_ids

            self._do_filter_validation(filter_text, gt, gte, lt, lte, et, boolean)

            def _is_valid(value):
                if gt and not value > gt:
                    return False
                if gte and not value >= gte:
                    return False
                if lt and not value < lt:
                    return False
                if lte and not value <= lte:
                    return False
                if et and not value != et:
                    return False
                return True

            if self.data_type in ["number"]:

                def is_valid(value):
                    return _is_valid(value)

            elif self.data_type in ["date"]:

                def is_valid(value):
                    return _is_valid(abs(self.date_scorer(value)))  # returns number of days

            elif self.data_type in ["bool"]:

                def is_valid(value):
                    return value == boolean

            elif self.data_type in ["string"]:
                # different from Elasticsearch filtering on strings, this method only allows
                #   exact presence of that input text and not a fuzzy match!

                filter_text = self._validate_and_reformat_value(self.data_type, filter_text)
                filter_text_aliases = {filter_text, filter_text.lower()}

                def is_valid(value):
                    if value in filter_text_aliases:
                        return True
                    if value.lower() in filter_text_aliases:
                        return True
                    return False

            allowed_ids = {
                _id: None
                for _id in allowed_ids
                if _id in self.id2value and is_valid(self.id2value[_id])
            }

            return allowed_ids

        def do_sort(self, curated_docs, sort_type, location=None):
            if not curated_docs:
                return curated_docs

            self._do_sort_validation(sort_type, location)

            validated_curated_docs, field_values = [], []
            for doc in curated_docs:
                _id = doc["_id"]
                value = self.id2value.get(_id)
                if value is None:
                    msg = (
                        f"Discarding doc with id: {doc['id']} while sorting as no "
                        f"{self.field_name} field available for it."
                    )
                    logger.info(msg)
                    continue
                validated_curated_docs.append(doc)
                field_values.append(value)
            curated_docs = validated_curated_docs

            if self.data_type == "location":
                sort_type = "asc"
                field_values = [self.location_scorer(value, location) for value in field_values]
            elif self.data_type == "number":
                field_values = [self.number_scorer(value) for value in field_values]
            elif self.data_type == "date":
                field_values = [self.date_scorer(value) for value in field_values]

            _, results = zip(
                *sorted(
                    enumerate(curated_docs),
                    key=lambda x: field_values[x[0]],
                    reverse=sort_type != "asc",
                )
            )
            return results

        def _do_search_validation(self, query_type):
            if self.data_type not in ["string", "date"]:
                msg = f"Searching is not allowed for data type '{self.data_type}'. "
                logger.error(msg)
                raise ValueError(
                    "Query can only be defined on text and vector fields. If it is,"
                    " try running load_kb with clean=True and reinitializing your"
                    " QuestionAnswerer object."
                )

            if query_type not in ALL_QUERY_TYPES:
                msg = f"Unknown query type- {query_type}"
                logger.error(msg)
                raise ValueError(msg)

        def _do_filter_validation(self, filter_text, gt, gte, lt, lte, et, boolean):
            # code inspired and adapted from Elasticsearch QA class (see Sort/Filter/QueryClause)

            if self.data_type in ["number", "date"]:
                if not gt and not gte and not lt and not lte and not et:
                    raise ValueError("No range parameter is specified")
                if gte and gt:
                    raise ValueError(
                        "Invalid range parameters. Cannot specify both 'gte' and 'gt'."
                    )
                if lte and lt:
                    raise ValueError(
                        "Invalid range parameters. Cannot specify both 'lte' and 'lt'."
                    )
                if (gt or gte or lt or lte) and et:
                    raise ValueError(
                        "Invalid range parameters. Cannot specify 'et' when specifying one of "
                        "[gt, gte, lt, lte]"
                    )
            elif self.data_type in ["bool"]:
                if not isinstance(boolean, bool):
                    raise ValueError("Invalid boolean parameter.")
            elif self.data_type in ["string"]:
                if not self.is_string(filter_text):
                    raise ValueError("Invalid textual input parameter.")
            else:
                raise ValueError(
                    "Custom filter can only be defined for boolean, number, string or date field."
                )

        def _do_sort_validation(self, sort_type, location):
            # code inspired and adapted from Elasticsearch QA class (see Sort/Filter/QueryClause)

            if sort_type not in SORT_TYPES:
                raise ValueError("Invalid value for sort type '{}'".format(sort_type))

            if self.data_type == "location" and sort_type != SORT_DISTANCE:
                raise ValueError("Invalid value for sort type '{}'".format(sort_type))

            if self.data_type == "location" and not location:
                raise ValueError("No origin location specified for sorting by distance.")

            if sort_type == SORT_DISTANCE and self.data_type != "location":
                raise ValueError("Sort by distance is only supported using 'location' field.")

            if self.data_type not in ["number", "date", "location"]:
                raise ValueError(
                    "Custom sort criteria can only be defined for"
                    + " 'number', 'date' or 'location' fields."
                )

        @property
        def doc_ids(self):
            return [*self.id2value.keys()]

        @staticmethod
        def curate_docs_to_return(index_resources, _ids, _scores=None):
            """
            Collates all field names into docs

            Args:
                index_resources: a dict of field names and corresponding FieldResource instances
                _ids (List[str]): if provided as a list of strings, only docs with those ids are
                    obtained in the same order of the ids, else all ids are used
                _scores (List[number], optional): if provided as a list of numbers and of same size
                    as the _ids, they will be attached to the curated results for corresponding _ids
            Returns:
                list[dict]: compiled docs
            """
            docs = {}

            if _scores:
                if len(_scores) != len(_ids):
                    msg = (
                        f"Number of ids ({len(_ids)}) supplied did not match "
                        f"number of  scores ({len(_scores)}). Discarding inputted scores "
                        f"while curating docs for QA results. "
                    )
                    logger.warning(msg)
                else:
                    docs = {_id: {"_id": _id, "_score": _scores[i]} for i, _id in enumerate(_ids)}

            if not docs:
                # when no _scores are inputted or when number of _scores do not match umber of _ids
                docs = {_id: {"_id": _id} for _id in _ids}

            # populate docs for all collected ids
            for field_name, field_resource in index_resources.items():
                for _id in _ids:
                    try:
                        docs[_id][field_name] = field_resource.id2value[_id]
                    except KeyError:
                        docs[_id][field_name] = None

            return [*docs.values()]

        @classmethod
        def from_metadata(cls, cache_object: Bunch):
            """
            Creates a fit resource from metadata

            Args:
                cache_object (Bunch): a Bunch dictionary object with attribute names and values

            Returns:
                FieldResource
            """
            field_resource = cls(
                index_name=cache_object.index_name,
                field_name=cache_object.field_name,
            )
            field_resource.data_type = cache_object.data_type
            field_resource.id2value = cache_object.id2value
            field_resource.hash = cache_object.hash
            field_resource.processor_type = cache_object.processor_type
            field_resource.has_text_resolver = cache_object.has_text_resolver
            field_resource.has_embedding_resolver = cache_object.has_embedding_resolver
            return field_resource

        def to_metadata(self):
            """
            Returns a Bunch object consisting of various details about the resource- (scoped) index
            name, the name of the KB field for which the resource is built for, the field's data
            type, all the KB data associated with this field, a hash that uniquely identifies the
            data, the data processor type and the presence of text as well as embedder resolvers.
            The returned metadata object does not contain any fit entity resolvers.

            Returns:
                Bunch: various meta information of this field resource
            """
            cache_object = Bunch(
                index_name=self.index_name,
                field_name=self.field_name,
                data_type=self.data_type,
                id2value=self.id2value,
                hash=self.hash,
                processor_type=self.processor_type,
                has_text_resolver=self.has_text_resolver,
                has_embedding_resolver=self.has_embedding_resolver,
            )
            return cache_object

    class Search:
        """Search class enabling functionality to query, filter and sort.
        Utilizes various methods from Indices and FiledResource to compute results.

        Currently, the following are supported data types for each clause type:
        Query -> "string", "date"  (assumes that "date" field exists as strings, both are supported
                    through kwargs)
        Filter -> "number" and "date" (through range parameters), "bool" (though boolean parameter),
                    "string" (through kwargs)
        Sort -> "number" and "date" (by specifying sort_type=asc or sort_type=desc),
                    "location" (by specifying sort_type=distance and passing origin
                    'location' parameter)

        Note: This Search class supports more items than Elasticsearch based QA

        """

        def __init__(self, index):
            """Initialize a Search object."""
            self.index_name = index
            self._search_queries = {}
            self._filter_queries = {}
            self._sort_queries = {}

        def query(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
            for field, value in kwargs.items():
                if field in self._search_queries:
                    msg = (
                        f"Found a duplicate search clause against '{field}' field name. "
                        "Utilizing only latest input."
                    )
                    logger.warning(msg)
                self._search_queries.update({field: {"query_type": query_type, "value": value}})

            return self

        def filter(self, query_type=DEFAULT_QUERY_TYPE, **kwargs):
            # Note: 'query_type' only kept to maintain similar arguments as ES based QA
            if query_type:
                query_type = None

            field = kwargs.pop("field", None)
            gt = kwargs.pop("gt", None)
            gte = kwargs.pop("gte", None)
            lt = kwargs.pop("lt", None)
            lte = kwargs.pop("lte", None)
            et = kwargs.pop("et", None)  # NEW! support equals-to filter; similar to above
            boolean = kwargs.pop("boolean", None)  # NEW! support boolean filter; input True/False

            # filter that operates on numeric values or date values or boolean
            if field:
                if field in self._filter_queries:
                    msg = (
                        f"Found a duplicate filter clause against '{field}' field name. "
                        "Utilizing only latest input."
                    )
                    logger.warning(msg)
                self._filter_queries.update(
                    {
                        field: {
                            "gt": gt,
                            "gte": gte,
                            "lt": lt,
                            "lte": lte,
                            "et": et,
                            "boolean": boolean,
                        }
                    }
                )

            # filter that operates on strings; extract field name and query strings then
            else:
                for key, filter_text in kwargs.items():
                    if key in self._filter_queries:
                        msg = (
                            f"Found a duplicate filter clause against '{key}' field name. "
                            "Utilizing only latest input."
                        )
                        logger.warning(msg)
                    self._filter_queries.update({key: {"filter_text": filter_text}})

            return self

        def sort(self, field, sort_type=None, location=None):
            if field in self._sort_queries:
                msg = (
                    f"Found a duplicate sort clause against '{field}' field name. "
                    "Utilizing only latest input."
                )
                logger.warning(msg)
            self._sort_queries.update({field: {"sort_type": sort_type, "location": location}})

            return self

        def execute(self, size=10):
            try:
                # fetch all indexes
                index_resources = NativeQuestionAnswerer.ALL_INDICES.get_index(self.index_name)
                # obtain all doc ids in the order they were present in the KB
                index_all_ids = NativeQuestionAnswerer.ALL_INDICES.get_all_ids(self.index_name)
            except KeyError:
                msg = (
                    f"The index '{self.index_name}' looks unavailable. "
                    f"Consider running '.load_kb(...)' to create indices "
                    f"before creating search/filter/sort queries. "
                )
                logger.error(msg)
                return []

            def requires_field_resource(f, i):
                msg = f"The field '{f}' is not available in index '{i}'."
                logger.error(msg)
                raise ValueError("Invalid knowledge base field '{}'".format(f))

            allowed_ids = None

            # if any filter queries exist, filter the necassary ids to do further processing
            if self._filter_queries:
                # can't make it set; preserve order
                allowed_ids = {_id: None for _id in index_all_ids}
                # get narrowed results for 'filter' clause
                for field_name, kwargs in self._filter_queries.items():
                    field_resource = index_resources.get(field_name)
                    if not field_resource:
                        requires_field_resource(field_name, self.index_name)
                    allowed_ids = field_resource.do_filter(allowed_ids, **kwargs)
                # return if nothing to process in further steps
                if not allowed_ids:
                    return []

            # get results (aka. curated_docs) for 'query' clauses, in decreasing order of similarity
            n_scores = 0
            scores = {}
            for field_name, kwargs in self._search_queries.items():
                query_type = kwargs["query_type"]
                value = kwargs["value"]
                field_resource = index_resources.get(field_name)
                if not field_resource:
                    requires_field_resource(field_name, self.index_name)
                this_field_scores = field_resource.do_search(query_type, value, allowed_ids)
                scores = {
                    _id: (scores.get(_id, 0.0) + _score) / (n_scores + 1)
                    for _id, _score in this_field_scores.items()
                }
                n_scores += 1
            # pass in all indices if no similarities computed
            if scores:
                _ids, _scores = zip(*sorted(scores.items(), key=lambda x: x[1], reverse=True))
            else:
                _ids, _scores = allowed_ids or index_all_ids, None
            has_similarity_scores = _scores is not None
            curated_docs = NativeQuestionAnswerer.FieldResource.curate_docs_to_return(
                index_resources, _ids=_ids, _scores=_scores
            )

            # if sim scores are available, get the top_n and then sort only the top_n objects
            if has_similarity_scores:
                if len(curated_docs) > size:
                    docs_scores = np.array([x["_score"] for x in curated_docs])
                    n_scores = len(docs_scores)
                    top_inds = docs_scores.argpartition(n_scores - size)[-size:]
                    curated_docs = [curated_docs[i] for i in top_inds]
                curated_docs = sorted(curated_docs, key=lambda x: x["_score"], reverse=True)[:size]

            # get sorted results for 'sort' clause, only on the resultant curated_docs
            for field_name, kwargs in self._sort_queries.items():
                field_resource = index_resources.get(field_name)
                if not field_resource:
                    requires_field_resource(field_name, self.index_name)
                curated_docs = field_resource.do_sort(curated_docs, **kwargs)

            curated_docs = curated_docs[:size]
            if len(curated_docs) < size:
                msg = (
                    f"Retrieved only {len(curated_docs)} matches instead of asked number "
                    f"{size} for index '{self.index_name}'."
                )
                logger.info(msg)

            # remove '_id' key field, as it meant for internal purposes only!
            # not removing '_score' key field, as user might need that for further analysis
            for doc in curated_docs:
                doc.pop("_id", None)

            return curated_docs

    @classmethod
    def _unload_all_indices(cls):
        NativeQuestionAnswerer.ALL_INDICES = NativeQuestionAnswerer.Indices(
            app_path=DEFAULT_APP_PATH
        )

    ALL_INDICES = Indices(app_path=DEFAULT_APP_PATH)


# TODO: Both QA classes assume input text is in English.
#  Going forward, this should be configurable and multilingual support should be added!
QuestionAnswererFactory.register_question_answerer_class("native", NativeQuestionAnswerer)
