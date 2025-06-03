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
import string
import random
import os
import zipfile
from typing import Iterable, Tuple
from urllib.request import urlretrieve

import torch

from .components._util import _get_module_or_attr
from .components._config import ENGLISH_LANGUAGE_CODE
from .augmentor_base import register_augmentor, Augmentor, UnsupportedLanguageError
from .markup import dump_query
from .core import Entity, Span, QueryEntity, ProcessedQuery, _get_overlap
from .path import (
    EMBEDDINGS_FOLDER_PATH,
    PARAPHRASER_FILE_PATH,
    PARAPHRASER_MODEL_PATH,
    HUGGINGFACE_PARAPHRASER_MODEL_PATH,
)
from .resource_loader import ResourceLoader
from .models.containers import TqdmUpTo

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGE_CODES = ["en", "es", "fr", "it", "pt", "ro"]
EOS_TOKEN = "</s>"
DEFAULT_NUM_PARAPHRASES = 10
PARAPHRASER_RETAIN_ENTITIES_URL = (
    "https://mindmeld-binaries.s3.amazonaws.com/paraphraser/paraphrase_retain_entities.zip"
)


class EnglishParaphraser(Augmentor):
    """Paraphraser class for generating English paraphrases."""

    def __init__(
        self,
        batch_size: int,
        language: str,
        retain_entities: bool,
        paths: Iterable[str],
        path_suffix: str,
        resource_loader: ResourceLoader,
    ):
        """Initializes an English paraphraser.

        Args:
            batch_size (int): Batch size for batch processing.
            language (str): The language code for paraphrasing.
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """

        if language != ENGLISH_LANGUAGE_CODE:
            raise UnsupportedLanguageError(
                f"'{language}' is not supported by the English Augmentor class"
            )

        super().__init__(
            language=language,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        pegasus_tokenizer = _get_module_or_attr("transformers", "PegasusTokenizer")
        pegasus_for_conditional_generation = _get_module_or_attr(
            "transformers", "PegasusForConditionalGeneration"
        )
        self.retain_entities = retain_entities
        if self.retain_entities:
            if not os.path.exists(PARAPHRASER_MODEL_PATH):
                self._download_model()
            model_name = PARAPHRASER_MODEL_PATH
        else:
            model_name = HUGGINGFACE_PARAPHRASER_MODEL_PATH
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = pegasus_tokenizer.from_pretrained(model_name)
        self.model = pegasus_for_conditional_generation.from_pretrained(model_name).to(
            self.torch_device
        )
        self.model.eval()

        # Update default params with user model config
        self.batch_size = batch_size

        self.default_paraphraser_model_params = {
            "max_length": 60,
            "num_beams": DEFAULT_NUM_PARAPHRASES,
            "num_return_sequences": DEFAULT_NUM_PARAPHRASES,
            "temperature": 1.5,
        }

        self.default_tokenizer_params = {
            "truncation": True,
            "padding": "longest",
            "max_length": 60,
        }

    def _download_model(self):
        logger.info(
            "Downloading paraphrase model from %s",
            PARAPHRASER_RETAIN_ENTITIES_URL,
        )

        # Make the folder that will contain the model folder
        if not os.path.exists(EMBEDDINGS_FOLDER_PATH):
            os.makedirs(EMBEDDINGS_FOLDER_PATH)

        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc="") as t:
            try:
                urlretrieve(
                    PARAPHRASER_RETAIN_ENTITIES_URL,
                    PARAPHRASER_FILE_PATH,
                    reporthook=t.update_to,
                )
            except ConnectionError as e:
                logger.error("Model download failed with error: %s", e)
                return
        try:
            with zipfile.ZipFile(PARAPHRASER_FILE_PATH, "r") as zip_ref:
                zip_ref.extractall(EMBEDDINGS_FOLDER_PATH)
            os.remove(PARAPHRASER_FILE_PATH)
        except zipfile.BadZipfile:
            logger.error("Unable to extract zip file. Try downloading the model again.")

    def _prepare_inputs(self, processed_queries: Iterable[ProcessedQuery]):
        """Processes input as expected by the two different English models
        Example:
            The default model requires just <unannotated text>
            The retain_entities model requires <unannotated text> <EOS> <entity values>

        Args:
            processed queries (list(ProcessedQuery)): List of ProcessedQuery objects from the app

        Return:
            model_inputs (list(str)): List of queries to be paraphrased in the
                                      format required by the models
            processed_queries (list(ProcessedQuery)): List of ProcessedQuery
                                                       with entity annotations
        """
        model_inputs = []
        for processed_query in processed_queries:
            processed_query_text = processed_query.query.text.strip()
            if self.retain_entities:
                text = [processed_query_text.lower(), EOS_TOKEN]
                for entity in processed_query.entities:
                    text.append(entity.text.lower())
                model_inputs.append(" ".join(text))
            else:
                model_inputs.append(processed_query_text)
        return model_inputs

    def _replace_with_random_gaz_entity(
        self, paraphrase_text: str, entity_matches: Iterable[Tuple[Entity, Span]]
    ):
        """Replaces values of annotated entities with randomly sampled ones from gazetteers

        Args:
            paraphrase_text (str): The paraphrased unannotated text
            entity_matches (List((Entity,Span))): List of (Entity, Span) values
                                                  found in the paraphrase_text

        Return:
            processed paraphrases (ProcessedQuery): ProcessedQuery of the paraphrase_text
        """
        new_paraphrase_text = []
        # Start replacing entities in ascending order of span starts
        entity_matches.sort(key=lambda x: x[1].start, reverse=False)
        running_start = 0
        previous_end = 0
        replaced_spans_entities = []
        for entity, span in entity_matches:
            # Calculate new start based on previously replaced entity length
            # For the first entity in the query, previous_end will be 0
            running_start += span.start - previous_end
            # Append text seen between entities
            new_paraphrase_text.append(paraphrase_text[previous_end : span.start])
            gaz = None
            # If not a system entity and gazetteer is available, load it
            if not Entity.is_system_entity(entity.type):
                gaz = self._resource_loader.get_gazetteer(entity.type)["entities"]
            if gaz:
                # Create new Entity and Span based on random gaz entry for entity type
                random_gaz_entity_text = random.sample(gaz, 1)[0]
                new_span = Span(
                    start=running_start,
                    end=running_start + len(random_gaz_entity_text) - 1,
                )
                new_entity = Entity(
                    text=random_gaz_entity_text,
                    entity_type=entity.type,
                    role=entity.role,
                    value=None,
                )
            else:
                new_entity = entity
                new_span = Span(
                    start=running_start,
                    end=running_start + len(entity.text) - 1,
                )
            running_start += len(new_entity.text)
            replaced_spans_entities.append([new_entity, new_span])
            new_paraphrase_text.append(new_entity.text)
            previous_end = span.end + 1
        # Append any leftover text
        new_paraphrase_text.append(paraphrase_text[previous_end:])

        processed_query = self._resource_loader.query_factory.create_query(
            "".join(new_paraphrase_text)
        )
        final_entities = [
            QueryEntity.from_query(query=processed_query, span=span, entity=entity)
            for (entity, span) in replaced_spans_entities
        ]
        return ProcessedQuery(query=processed_query, entities=tuple(final_entities))

    def _annotate_entities(
        self, paraphrases: Iterable[str], processed_queries: Iterable[ProcessedQuery]
    ):
        """Annotates entities in the generated paraphrases with the entities in the original
        query

        Args:
            paraphrases (list(str)): List of unannotated paraphrases of queries
            processed_queries (list(ProcessedQuery)): List of their corresponding original
                ProcessedQuery

        Return:
            paraphrases (list(str)): List of paraphrased queries.
        """
        valid_paraphrases = []
        for i, processed_query in enumerate(processed_queries):
            # sort entities so we annotate the longest one first
            entities = sorted(
                list(processed_query.entities),
                key=lambda x: len(x.text),
                reverse=True,
            )
            # fetch paraphrases for the query from the batch
            queries = paraphrases[
                (i * DEFAULT_NUM_PARAPHRASES) : (i * DEFAULT_NUM_PARAPHRASES)
                + DEFAULT_NUM_PARAPHRASES
            ]
            for query in queries:
                if not query:
                    continue
                all_matches = []
                for entity in entities:
                    found_matches = re.finditer(entity.text.lower(), query)
                    for match in found_matches:
                        matched_span = Span(start=match.start(0), end=match.end(0) - 1)
                        matched_entity = Entity(
                            text=match.group(0),
                            entity_type=entity.entity.type,
                            role=entity.entity.role,
                            value=None,
                        )
                        # check if found entity has no overlaps with previously matched entities
                        no_overlaps = [not _get_overlap(m[1], matched_span) for m in all_matches]
                        if all(no_overlaps):
                            all_matches.append((matched_entity, matched_span))
                # We are taking a call here to only return paraphrases that contain all entities
                # that were present in the original query
                if len(all_matches) == len(entities):
                    processed_paraphrase = self._replace_with_random_gaz_entity(query, all_matches)
                    # Dump the processed paraphrase queries in the mindmeld markdown format
                    valid_paraphrases.append(dump_query(processed_paraphrase))
        return valid_paraphrases

    @staticmethod
    def _normalize_paraphrases(queries: Iterable[str]) -> Iterable[str]:
        # This function removes punctuations since these generative models
        # have a tendency to repeat them.
        # Since most classifiers use normalized text, this should not be an issue.
        without_puncts = [
            s.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            for s in queries
        ]
        queries = [" ".join(s.split()) for s in without_puncts if s]
        return queries

    def _generate_paraphrases(self, processed_queries: Iterable[str]) -> Iterable[str]:
        """Generates paraphrase responses for given query.

        Args:
            queries (list(str)): List of application queries.

        Return:
            paraphrases (list(str)): List of paraphrased queries.
        """
        all_generated_queries = []

        for pos in range(0, len(processed_queries), self.batch_size):
            processed_input_queries = processed_queries[pos : pos + self.batch_size]
            tokenizer_input = self._prepare_inputs(processed_input_queries)
            batch = self.tokenizer.prepare_seq2seq_batch(
                tokenizer_input,
                **self.default_tokenizer_params,
                return_tensors="pt",
            ).to(self.torch_device)
            with torch.no_grad():
                generated = self.model.generate(
                    **batch,
                    **self.default_paraphraser_model_params,
                )
            decoded_queries = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if self.retain_entities:
                decoded_queries = self._normalize_paraphrases(decoded_queries)
                decoded_queries = self._annotate_entities(decoded_queries, processed_input_queries)
            all_generated_queries.extend(decoded_queries)
        return all_generated_queries

    def augment_queries(self, processed_queries: Iterable[str], **kwargs):
        augmented_queries = list(
            set(
                p.lower()
                for p in self._generate_paraphrases(processed_queries, **kwargs)
                if self._validate_generated_query(p)
            )
        )
        return augmented_queries


class MultiLingualParaphraser(Augmentor):
    """Paraphraser class for generating paraphrases based on language code of the app
    (currently supports: French, Italian, Portuguese, Romanian and Spanish).
    """

    def __init__(
        self,
        batch_size: int,
        language: str,
        retain_entities: bool,
        paths: Iterable[str],
        path_suffix: str,
        resource_loader: ResourceLoader,
    ):
        """Initializes a multi-lingual paraphraser.

        Args:
            batch_size (int): Batch size for batch processing.
            language (str): The language code for paraphrasing.
            paths (list): Path rules for fetching relevant files to Paraphrase.
            path_suffix (str): Suffix to be added to new augmented files.
            resource_loader (object): Resource Loader object for the application.
        """
        if language not in ["es", "fr", "it", "pt", "ro"]:
            raise UnsupportedLanguageError(
                f"'{language}' is not supported by the MultiLingual Augmentor class"
            )

        super().__init__(
            language=language,
            paths=paths,
            path_suffix=path_suffix,
            resource_loader=resource_loader,
        )

        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retain_entities = retain_entities

        marian_tokenizer = _get_module_or_attr("transformers", "MarianTokenizer")
        marian_mt_model = _get_module_or_attr("transformers", "MarianMTModel")

        en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
        self.en_tokenizer = marian_tokenizer.from_pretrained(en_model_name)
        self.en_model = marian_mt_model.from_pretrained(en_model_name)
        self.en_model.to(self.torch_device)
        self.en_model.eval()

        target_model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        self.target_tokenizer = marian_tokenizer.from_pretrained(target_model_name)
        self.target_model = marian_mt_model.from_pretrained(target_model_name).to(self.torch_device)
        self.target_model.eval()

        # Update default params with user model config
        self.batch_size = batch_size

        self.default_forward_params = {
            "max_length": 60,
            "num_beams": 5,
            "num_return_sequences": 5,
            "temperature": 1.0,
            "top_k": 0,
        }

        self.default_reverse_params = {
            "max_length": 60,
            "num_beams": 3,
            "num_return_sequences": 3,
            "temperature": 1.0,
            "top_k": 0,
        }

    def _translate(self, *, queries: Iterable[str], model, tokenizer, **kwargs) -> Iterable[str]:
        """The core translation step for forward and reverse translation.

        Args:
            template (lambda func): Structure input text to model.
            queries (list(str)): List of input queries.
            model: Machine translation model (en-ROMANCE or ROMANCE-en).
            tokenizer: Language tokenizer for input query text.
        """
        all_translated_queries = []
        for pos in range(0, len(queries), self.batch_size):
            encoded = tokenizer.prepare_seq2seq_batch(
                queries[pos : pos + self.batch_size], return_tensors="pt"
            ).to(self.torch_device)
            for key in encoded:
                encoded[key] = encoded[key].to(self.torch_device)
            with torch.no_grad():
                translated = model.generate(**encoded, **kwargs)
            translated_queries = tokenizer.batch_decode(translated, skip_special_tokens=True)
            all_translated_queries.extend(translated_queries)
        return all_translated_queries

    def _prepare_inputs(self, processed_queries: Iterable[ProcessedQuery]) -> Iterable[str]:
        """Removes any markdown formatting in the query

        Args:
            queries (list(ProcessedQuery)): List of queries to be paraphrased

        Returns:
            unannotated queries (list(str))

        """
        unannotated_queries = [
            processed_query.query.text.strip() for processed_query in processed_queries
        ]
        return unannotated_queries

    def augment_queries(self, processed_queries: Iterable[ProcessedQuery]) -> Iterable[str]:
        translated_queries = self._translate(
            queries=self._prepare_inputs(processed_queries),
            model=self.en_model,
            tokenizer=self.en_tokenizer,
            **self.default_forward_params,
        )

        def template(text):
            return f">>{self.language_code}<< {text}"

        translated_queries = [template(query) for query in set(translated_queries)]

        reverse_translated_queries = self._translate(
            queries=translated_queries,
            model=self.target_model,
            tokenizer=self.target_tokenizer,
            **self.default_reverse_params,
        )
        augmented_queries = list(
            set(p.lower() for p in reverse_translated_queries if self._validate_generated_query(p))
        )

        return augmented_queries


register_augmentor("EnglishParaphraser", EnglishParaphraser)
register_augmentor("MultiLingualParaphraser", MultiLingualParaphraser)
