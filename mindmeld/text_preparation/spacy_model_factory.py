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

"""This module contains a Spacy Model Factory."""
import importlib
import logging
import subprocess
from typing import Any, Iterable

import spacy
from spacy.symbols import ORTH, NORM, IS_CURRENCY

from ..constants import (
    SPACY_WEB_TRAINED_LANGUAGES,
    SPACY_SUPPORTED_LANGUAGES,
    SPACY_MODEL_SIZES,
)

logger = logging.getLogger(__name__)


class SpacyModelFactory:
    """Spacy (Language) Model Factory Class"""

    @staticmethod
    def get_spacy_language_model(language: str, spacy_model_size="lg", disable: Iterable[str] = ()):
        """Get a Spacy Language model.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
            disable (Iterable[str]): Tuple of pipeline elements to disable. ('ner', 'tagger',
                'parser', etc.)

        Returns:
            nlp: Spacy language model. (Ex: "spacy.lang.es.Spanish")
        """
        SpacyModelFactory.validate_spacy_language(language)
        SpacyModelFactory.validate_spacy_model_size(spacy_model_size)
        spacy_model_name = SpacyModelFactory._get_spacy_model_name(language, spacy_model_size)
        return SpacyModelFactory._load_model(spacy_model_name, disable)

    @staticmethod
    def validate_spacy_language(language: str) -> None:
        """Check if the language is valid.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
        """
        if language not in SPACY_SUPPORTED_LANGUAGES:
            raise ValueError("Spacy does not currently support: {!r}.".format(language))

    @staticmethod
    def validate_spacy_model_size(spacy_model_size: str) -> None:
        """Check if the model size is valid.

        Args:
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")
        """
        if spacy_model_size not in SPACY_MODEL_SIZES:
            raise ValueError(
                "{!r} is not a valid model size. Select from: {!r}.".format(
                    spacy_model_size, " ".join(SPACY_MODEL_SIZES)
                )
            )

    @staticmethod
    def _add_special_cases(nlp):
        """Add special cases to the model."""
        # The default language model only infers currency symbols and not the actual *names*
        # of currency. The only values included here are words without multiple meanings
        # as I'm not sure whether the tokenizer special case will end up obscuring other
        # possible definitions (i.e. "pounds" isn't included)
        # Also note that internally all currency symbols are normalized to "$" so we are
        # actually matching the existing SpaCy behavior.
        for currency in ["euro", "dollar"]:
            nlp.tokenizer.add_special_case(currency, [{ORTH: currency, NORM: "$"}])
            currency += "s"
            nlp.tokenizer.add_special_case(currency, [{ORTH: currency, NORM: "$"}])

    @staticmethod
    def _load_model(spacy_model_name: str, disable: Iterable[str] = ()) -> spacy.Language:
        """Load Spacy English model. Download if needed.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
            disable (Iterable[str]): Tuple of pipeline elements to disable. ('ner', 'tagger',
                'parser', etc.)

        Returns:
            nlp: Spacy language model. (Ex: "spacy.lang.es.Spanish")
        """
        logger.info("Loading Spacy model %s.", spacy_model_name)
        try:
            nlp = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning("%s not found on disk. Downloading the model.", spacy_model_name)
            SpacyModelFactory._download_spacy_model(spacy_model_name)
            language_module = SpacyModelFactory._import_spacy_model(spacy_model_name)
            nlp = language_module.load(disable=disable)

        SpacyModelFactory._add_special_cases(nlp)
        return nlp

    @staticmethod
    def _get_spacy_model_name(language: str, spacy_model_size: str) -> str:
        """Get the name of a Spacy Model.

        Args:
            language (str, optional): Language as specified using a 639-1/2 code.
            spacy_model_size (str, optional): Size of the Spacy model to use. ("sm", "md", or "lg")

        Returns:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
        """
        model_type = "web" if language in SPACY_WEB_TRAINED_LANGUAGES else "news"
        return f"{language}_core_{model_type}_{spacy_model_size}"

    @staticmethod
    def _download_spacy_model(spacy_model_name: str) -> None:
        """Download Spacy Model.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")
        """
        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "download",
                spacy_model_name,
                "--default-timeout=1000",
            ],
            check=True,
        )

    @staticmethod
    def _import_spacy_model(spacy_model_name: str) -> Any:
        """Attempt to Imort the Spacy Model.

        Args:
            spacy_model_name (str): Name of the Spacy NER model (Ex: "en_core_web_sm")

        Returns:
            language_module (module): Imported language module.
        """
        try:
            return importlib.import_module(spacy_model_name)
        except ModuleNotFoundError as error:
            raise ValueError("Unknown Spacy model name: {!r}.".format(spacy_model_name)) from error
