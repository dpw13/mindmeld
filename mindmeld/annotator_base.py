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
import logging
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, List, Type
from tqdm import tqdm
from .resource_loader import ResourceLoader
from .components._config import (
    ENGLISH_LANGUAGE_CODE,
    ENGLISH_US_LOCALE,
)
from .system_entity_recognizer import (
    DucklingRecognizer,
)
from .markup import load_query, dump_queries
from .core import (
    Entity,
    Span,
    ProcessedQuery,
    QueryEntity,
    _get_overlap,
)
from .exceptions import MarkupError
from .query_factory import QueryFactory

logger = logging.getLogger(__name__)


class AnnotatorAction(Enum):
    ANNOTATE = "annotate"
    UNANNOTATE = "unannotate"


class Annotator(ABC):
    """
    Abstract Annotator class that can be used to build a custom Annotation class.
    """

    # pylint: disable=W0613
    def __init__(
        self,
        app_path,
        annotation_rules=None,
        language=ENGLISH_LANGUAGE_CODE,
        locale=ENGLISH_US_LOCALE,
        overwrite=False,
        unannotate_supported_entities_only=True,
        unannotation_rules=None,
        **kwargs,
    ):
        """Initializes an annotator.

        Args:
            app_path (str): The location of the MindMeld app.
            annotation_rules (list): List of Annotation rules.
            language (str, optional): Language as specified using a 639-1/2 code.
            locale (str, optional): The locale representing the ISO 639-1 language code and \
                ISO3166 alpha 2 country code separated by an underscore character.
            overwrite (bool): Whether to overwrite existing annotations with conflicting spans.
            unannotate_supported_entities_only (bool): Only allow removal of supported entities.
            unannotation_rules (list): List of Annotation rules.
        """
        self.app_path = app_path
        self.language = language
        self.locale = locale
        self.overwrite = overwrite
        self.annotation_rules = annotation_rules or []
        self.unannotate_supported_entities_only = unannotate_supported_entities_only
        self.unannotation_rules = unannotation_rules or []
        self._resource_loader = ResourceLoader.create_resource_loader(app_path)
        self.duckling = DucklingRecognizer.get_instance()

    def _get_file_entities_map(self, action: AnnotatorAction):
        """Creates a dictionary that maps file paths to entities given
        regex rules defined in the config.

        Args:
            action (AnnotatorAction): Can be "annotate" or "unannotate". Used as a key
                to access a list of regex rules in the config dictionary.

        Returns:
            file_entities_map (dict): A dictionary that maps file paths in an
                App to a list of entities.
        """
        all_file_paths = self._resource_loader.get_all_file_paths()
        file_entities_map = {path: [] for path in all_file_paths}

        if action == AnnotatorAction.ANNOTATE:
            rules = self.annotation_rules
        elif action == AnnotatorAction.UNANNOTATE:
            rules = self.unannotation_rules
        else:
            raise AssertionError(f"{action} is an invalid Annotator action.")

        for rule in rules:
            pattern = Annotator._get_pattern(rule)
            compiled_pattern = re.compile(pattern)
            filtered_paths = self._resource_loader.filter_file_paths(
                compiled_pattern=compiled_pattern, file_paths=all_file_paths
            )
            for path in filtered_paths:
                entities = self._get_entities(rule)
                file_entities_map[path] = entities
        return file_entities_map

    @staticmethod
    def _get_pattern(rule: Dict[str, str]) -> str:
        """Convert a rule represented as a dictionary with the keys "domains", "intents",
        "entities" into a regex pattern.

        Args:
            rule (dict): Annotation/Unannotation rule.

        Returns:
            pattern (str): Regex pattern specifying allowed file paths.
        """
        pattern = [rule[x] for x in ["domains", "intents", "files"]]
        return ".*/" + "/".join(pattern)

    def _get_entities(self, rule: Dict[str, str]) -> List[str]:
        """Process the entities specified in a rule dictionary. Check if they are valid
        for the given annotator.

        Args:
            rule (dict): Annotation/Unannotation rule with an "entities" key.

        Returns:
            valid_entities (list): List of valid entities specified in the rule.
        """
        if rule["entities"].strip() in ["*", ".*", ".+"]:
            return ["*"]
        entities = re.sub(r"[()]", "", rule["entities"]).split("|")
        valid_entities = []
        for entity in entities:
            entity = entity.strip()
            if self.valid_entity_check(entity):
                valid_entities.append(entity)
            else:
                logger.warning("%s is not a valid entity. Skipping entity.", entity)
        return valid_entities

    @property
    @abstractmethod
    def supported_entity_types(self) -> Iterable[str]:
        """
        Returns:
            supported_entity_types (list): List of supported entity types.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def valid_entity_check(self, entity: str) -> bool:
        """Determine if an entity type is valid.

        Args:
            entity (str): Name of entity to annotate.

        Returns:
            bool: Whether entity is valid.
        """
        entity = entity.lower().strip()
        return entity in self.supported_entity_types

    def annotate(self) -> None:
        """Annotate data."""
        if not self.annotation_rules:
            logger.warning(
                """'annotate' field is not configured or misconfigured in the `config.py`.
                 We can't find any file to annotate."""
            )
            return
        self._modify_queries(action=AnnotatorAction.ANNOTATE)

    def unannotate(self) -> None:
        """Unannotate data."""
        if not self.unannotate:
            logger.warning(
                """'unannotate' field is not configured or misconfigured in the `config.py`.
                 We can't find any file to unannotate."""
            )
            return
        self._modify_queries(action=AnnotatorAction.UNANNOTATE)

    def _modify_queries(self, action: AnnotatorAction):
        """Iterates through App files and annotates or unannotates queries.

        Args:
            action (AnnotatorAction): Can be "annotate" or "unannotate".
        """
        file_entities_map = self._get_file_entities_map(action=action)
        query_factory = QueryFactory.create_query_factory(self.app_path)
        path_list = [p for p in file_entities_map if file_entities_map[p]]
        for path in path_list:
            processed_queries = Annotator._get_processed_queries(
                file_path=path, query_factory=query_factory
            )
            tqdm_desc = "Processing " + path + ": "
            for processed_query in tqdm(processed_queries, ascii=True, desc=tqdm_desc):
                entity_types = file_entities_map[path]
                if action == AnnotatorAction.ANNOTATE:
                    self._annotate_query(
                        processed_query=processed_query,
                        entity_types=entity_types,
                    )
                elif action == AnnotatorAction.UNANNOTATE:
                    self._unannotate_query(
                        processed_query=processed_query,
                        remove_entities=entity_types,
                    )
            with open(path, "w") as outfile:
                outfile.write("".join(list(dump_queries(processed_queries))))
                outfile.close()

    @staticmethod
    def _get_processed_queries(file_path: str, query_factory: QueryFactory) -> List[ProcessedQuery]:
        """Converts queries in a given path to processed queries.
        Skips and presents a warning if loading the query creates an error.

        Args:
            file_path (str): Path to file containing queries.
            query_factory (QueryFactory): Used to generate processed queries.

        Returns:
            processed_queries (list): List of processed queries from file.
        """
        with open(file_path) as infile:
            queries = infile.readlines()
        processed_queries = []
        domain, intent = file_path.split(os.sep)[-3:-1]
        for query in queries:
            try:
                processed_query = load_query(
                    markup=query,
                    domain=domain,
                    intent=intent,
                    query_factory=query_factory,
                )
                processed_queries.append(processed_query)
            except (AssertionError, MarkupError):
                logger.warning("Skipping query. Error in processing: %s", query)
        return processed_queries

    def _annotate_query(self, processed_query: ProcessedQuery, entity_types: Iterable):
        """Updates the entities of a processed query with newly
        annotated entities.

        Args:
            processed_query (ProcessedQuery): The processed query to update.
            entity_types (list): List of entities allowed for annotation.
        """
        current_entities = list(processed_query.entities)
        annotated_entities = self._get_annotated_entities(
            processed_query=processed_query, entity_types=entity_types
        )
        final_entities = Annotator._resolve_conflicts(
            target_entities=annotated_entities if self.overwrite else current_entities,
            other_entities=current_entities if self.overwrite else annotated_entities,
        )
        processed_query.entities = tuple(final_entities)

    def _get_annotated_entities(self, processed_query: ProcessedQuery, entity_types=None) -> List:
        """Creates a list of query entities after parsing the text of a
        processed query.

        Args:
            processed_query (ProcessedQuery): A processed query.
            entity_types (list): List of entities allowed for annotation.

        Returns:
            query_entities (list): List of query entities.
        """
        if len(entity_types) == 0:
            return []
        entity_types = None if entity_types == ["*"] else entity_types
        return self.parse(
            sentence=processed_query.query.text,
            entity_types=entity_types,
            domain=processed_query.domain,
            intent=processed_query.intent,
        )

    @staticmethod
    def _item_to_query_entity(item: Dict[str, Any], processed_query: ProcessedQuery) -> QueryEntity:
        """Converts an item returned from parse into a query entity.

        Args:
            item (dict): Dictionary representing an entity with the keys -
                "body", "start", "end", "value", "dim". ("role" is an optional attribute.)
            processed_query (ProcessedQuery): The processed query that the
                entity is found in.

        Returns:
            query_entity (QueryEntity): The converted query entity.
        """
        span = Span(start=item["start"], end=item["end"] - 1)
        role = item.get("role")
        entity = Entity(
            text=item["body"],
            entity_type=item["dim"],
            role=role,
            value=item["value"],
        )
        query_entity = QueryEntity.from_query(query=processed_query.query, span=span, entity=entity)
        return query_entity

    @staticmethod
    def _resolve_conflicts(
        target_entities: List[QueryEntity],
        other_entities: Iterable[QueryEntity],
    ) -> Iterable[QueryEntity]:
        """Resolve overlaps between existing entities and newly annotad entities.

        Args:
            target_entities (list): List of existing query entities.
            other_entities (list): List of new query entities.

        Returns:
            final_entities (list): List of resolved query entities.
        """
        additional_entities = []
        for o_entity in other_entities:
            no_overlaps = [
                not _get_overlap(o_entity.span, t_entity.span) for t_entity in target_entities
            ]
            if all(no_overlaps):
                additional_entities.append(o_entity)
        target_entities.extend(additional_entities)
        return target_entities

    # pylint: disable=R0201
    def _unannotate_query(self, processed_query: ProcessedQuery, remove_entities: Iterable) -> None:
        """Removes specified entities in a processed query. If all entities are being
        removed, this function will not remove entities that the annotator does not support
        unless it is explicitly specified to do so in the config with the param
        "unannotate_supported_entities_only" (bool).

        Args:
            processed_query (ProcessedQuery): A processed query.
            remove_entities (list): List of entities to remove.
        """
        keep_entities = []
        for query_entity in processed_query.entities:
            if remove_entities == ["*"]:
                is_supported_entity = self.valid_entity_check(query_entity.entity.type)
                if self.unannotate_supported_entities_only and not is_supported_entity:
                    keep_entities.append(query_entity)
            elif query_entity.entity.type not in remove_entities:
                keep_entities.append(query_entity)
        processed_query.entities = tuple(keep_entities)

    @abstractmethod
    def parse(self, sentence: str, **kwargs) -> Iterable[QueryEntity]:
        """Extract entities from a sentence. Detected entities should be
        represented as dictionaries with the following keys: "body", "start"
        (start index), "end" (end index), "value", "dim" (entity type).

        Args:
            sentence (str): Sentence to detect entities.

        Returns:
            query_entities (list): List of QueryEntity objects.
        """
        raise NotImplementedError("Subclasses must implement this method")


ANNOTATOR_MAP: Dict[str, Type[Annotator]] = {}


def create_annotator(config: Dict) -> Annotator:
    """Creates an annotator instance using the provided configuration

    Args:
        config (dict): A model configuration

    Returns:
        Annotator: An Annotator class

    Raises:
        ValueError: When model configuration is invalid or required key is missing
    """
    if "annotator_class" not in config:
        raise KeyError("Missing required argument in AUTO_ANNOTATOR_CONFIG: 'annotator_class'")
    if config["annotator_class"] in ANNOTATOR_MAP:
        return ANNOTATOR_MAP[config.pop("annotator_class")](**config)
    else:
        msg = "Invalid model configuration: Unknown model type {!r}"
        raise KeyError(msg.format(config["annotator_class"]))


def register_annotator(annotator_class_name: str, annotator_class: Type[Annotator]) -> None:
    """Registers an Annotator class for use with `create_annotator()`

    Args:
        annotator_class_name (str): The annotator class name as specified in the config
        model_class (class): The annotator class to register
    """
    ANNOTATOR_MAP[annotator_class_name] = annotator_class
