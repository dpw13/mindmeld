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
This module contains the embedder model class.
"""
import json
import logging
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Callable, Type

import numpy as np

from ._util import (
    torch_op,
)
from .. import path
from ..resource_loader import Hasher

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """
    Base class for embedder model
    """

    class EmbeddingsCache:
        def __init__(self, cache_path=None):
            """
            Args:
                cache_path: A .pkl cache path to dump the embeddings cache
            """
            self.reset()
            if cache_path:
                self.load(self._get_cache_path(cache_path=cache_path))

        def reset(self):
            self.data = OrderedDict()

        def load(self, cache_path=None):
            """Loads the cache file."""

            cache_path = self._get_cache_path(cache_path)

            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                with open(cache_path, "rb") as fp:
                    data = pickle.load(fp)
                    fp.close()

                if (
                    "_texts" in data
                    and "_texts_embeddings" in data
                    and isinstance(data["_texts"], list)
                ):  # new format;
                    self.data = dict(zip(data["_texts"], data["_texts_embeddings"]))

                elif (
                    "synonyms" in data
                    and "synonyms_embs" in data
                    and isinstance(data["synonyms"], dict)
                ):  # deprecated format; backwards compatible with ER module code
                    self.data = {key: data["synonyms_embs"][j] for key, j in data["synonyms"]}

                else:  # deprecated format; backwards compatible with QA module code
                    if not isinstance(data, dict):
                        msg = (
                            "Unknown data format while loading cache embeddings. "
                            "Ignoring loading ..."
                        )
                        logger.error(msg)
                    self.data = data

        def clear(self, cache_path=None):
            """Deletes the cache file."""

            cache_path = self._get_cache_path(cache_path)

            if os.path.exists(cache_path):
                os.remove(cache_path)
                msg = f"Embedder cache cleared at {cache_path}"
                logger.info(msg)

        def dump(self, cache_path=None):
            """Dumps the cache to disk."""

            cache_path = self._get_cache_path(cache_path)

            if self.data:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                data = {
                    "_texts": [*self.data.keys()],
                    "_texts_embeddings": np.array([*self.data.values()]),
                }
                with open(cache_path, "wb") as fp:
                    pickle.dump(data, fp)
                    fp.close()

                msg = f"Embedder cache dumped at {cache_path}"
                logger.info(msg)

            else:
                msg = "No embedding data exists to dump. Ignoring dumping."
                logger.warning(msg)

        def get(self, text, default=None):
            return self.__getitem__(text, default)

        def _get_cache_path(self, cache_path):
            if not cache_path:
                msg = f"Invalid cache path '({cache_path})' provided for {self.__class__.__name__}."
                raise ValueError(msg)
            return os.path.abspath(cache_path)

        def __contains__(self, text):
            if text in self.data:
                return True
            return False

        def __getitem__(self, text, default=None):
            return self.data.get(text, default)

        def __setitem__(self, text, encoding):
            self.data[text] = encoding

        def __delitem__(self, text):
            try:
                del self.data[text]
            except KeyError as e:
                logger.error(e)
                pass

        def __iter__(self):
            # zip texts and encodings into a dictionary for iteration
            if self.data:
                return iter(self.data)

        def __len__(self):
            return len(self.data)

    def __init__(self, app_path=None, cache_path=None, **kwargs):
        """
        Initializes an embedder. The instantiated embedder model maintains a cache object that has
        embeddings of inputs observed so far through the .get_encodings() method. This cache can be
        useful especially if obtaining embeddings for the same text input is costlier in time versus
        a lookup.

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.
        """

        # load embedder model
        self.model = self.load()

        # obtain a cache path for creating an embedder cache object
        if cache_path is None:
            if app_path:
                deprecated_cache_path = path.get_embedder_cache_file_path(
                    app_path,
                    kwargs.get("embedder_type", "default"),
                    kwargs.get("model_name", "default"),
                )
                if (
                    os.path.exists(deprecated_cache_path)
                    and os.path.getsize(deprecated_cache_path) > 0
                ):
                    # deprecated usage:
                    #   Determine path from `embedder_type` and `model_name`
                    #   Inside a Mindmeld app, this path is generally something like:
                    #       '.generated/indexes/{embedder_type}_{model_name}_cache.pkl'
                    cache_path = deprecated_cache_path
                    msg = (
                        f"Found a deprecated cache path at '{cache_path}' that contains "
                        f"embeddings for a default configuration of embedder models. "
                        f"If you wish to use mindmeld version greater than 4.3.4 to work with "
                        f"non-default embedder configurations, consider deleting this cache "
                        f"path manually and run again."
                    )
                    logger.warning(msg)
                else:
                    # new usage:
                    #   Determine cache path for the model using `model_id`
                    #   Implies a previously used path name has no data and hence, is safe to change
                    #   default cache path for this model (backwards compatibility required only for
                    #   loading previously dumped embeddings data).
                    #   Cannot use previous path template because `model_name` alone is not
                    #   sufficient to uniquely identify a bert model (as it can be configured now).
                    #   Inside a Mindmeld app, this path is generally something like:
                    #       '.generated/indexes/{model_id}_cache.pkl'
                    cache_path = path.get_embedder_cache_file_path(app_path, self.model_id)
            else:
                msg = (
                    f"{self.__class__.__name__} embedder instantiated without a valid cache "
                    f"path. This will lead to an error if you try to dump the encodings cache. "
                    f"To have a valid cache dump location, pass-in 'app_path' or 'cache_path' "
                    f"argument. Alternatively, the `cache_path` can also be passed to the dump "
                    f"and load methods directly."
                )
                logger.info(msg)

        # load embedder cache object
        self.cache_path = cache_path
        self.cache = Embedder.EmbeddingsCache(self.cache_path)

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs"""
        msg = (
            "Embedder models need to have model ids to uniquely identify each model "
            "associated with a specific configuration. It can be set through the property "
            "setter 'model_id'. If unspecified, a default value ('default') is used instead."
        )
        logger.warning(msg)
        return "default"

    @abstractmethod
    def load(self, **kwargs):
        """Loads the embedder model

        Returns:
            The model object.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, text_list):
        """
        Args:
            text_list (list): A list of text strings for which to generate the embeddings.

        Returns:
            (list): A list of numpy arrays of the embeddings.
        """
        raise NotImplementedError

    def get_encodings(self, text_list, add_to_cache=True) -> List[Any]:
        """
        Fetches the encoded values from the cache, or generates them and adds to cache unless
        add_to_cache is set to False. This method is wrapped around .encode() by maintaining an
        embedding cache.

        Args:
            text_list (list): A list of text strings for which to get the embeddings.
            add_to_cache (bool): If True, adds the encodings to self.cache and returns embeddings

        Returns:
            (list): A list of numpy arrays with the embeddings.
        """

        uniques_text_list, uniques = [], {}
        text_list_to_uniques_text_list_map = []
        for text in text_list:
            if text not in uniques:
                uniques[text] = len(uniques)
                uniques_text_list.append(text)
            text_list_to_uniques_text_list_map.append(uniques[text])

        encoded = [self.cache.get(text, None) for text in uniques_text_list]
        cache_miss_indices = [i for i, vec in enumerate(encoded) if vec is None]
        text_to_encode = [uniques_text_list[i] for i in cache_miss_indices]
        model_encoded_text = self.encode(text_to_encode)

        for i, v in enumerate(cache_miss_indices):
            encoded[v] = model_encoded_text[i]
            if add_to_cache:
                self.cache[text_to_encode[i]] = model_encoded_text[i]

        return [encoded[text_list_to_uniques_text_list_map[i]] for i, text in enumerate(text_list)]

    def add_to_cache(self, mean_or_max_pooled_whitelist_embs):
        """
        Method to add custom embeddings to cache without triggering `.encode()`. Example, one can
        manually add some max-pooled or mean-pooled embeddings to cache. This method is created
        to entertain storing superficial text-encoding pairs (superficial because the encodings are
        not the encodings of the text itself but a combination of encodings of some list of texts
        from the same embedder model). For example, to add superficial entity embeddings as average
        of whitelist embeddings in Entity Resolution.

        Args:
            mean_or_max_pooled_whitelist_embs (dict): texts and their corresponding superficial
                embeddings as a 1D numpy array, having same length as emb_dim of the embedder
        """
        for key, value in mean_or_max_pooled_whitelist_embs.items():
            value = np.asarray(value).reshape(-1)
            known_emb_dim = getattr(self, "emb_dim", None)
            if known_emb_dim and not len(value) == known_emb_dim:
                msg = (
                    f"Expected superficial embedding of length {known_emb_dim} but found "
                    f"{len(value)}. Not adding the embedding for {key} to cache."
                )
                logger.error(msg)
            if key in self.cache:
                msg = f"Overwriting a superficial embedding for {key}"
                logger.warning(msg)
            self.cache[key] = value

    def dump_cache(self, cache_path=None):
        self.cache.dump(cache_path=cache_path or self.cache_path)

    def load_cache(self, cache_path=None):
        self.cache.load(cache_path=cache_path or self.cache_path)

    def clear_cache(self, cache_path=None):
        self.cache.clear(cache_path=cache_path or self.cache_path)

    def find_similarity(
        self,
        src_texts: List[str],
        tgt_texts: List[str] = None,
        top_n: int = 20,
        scores_normalizer: str = None,
        similarity_function: Callable[[List[Any], List[Any]], np.ndarray] = None,
        _return_as_dict=False,
        _no_sort=False,
    ):
        """Computes the cosine similarity

        Args:
            src_texts (Union[str, list]): string or list of strings to obtain matching scores for.
            tgt_texts (list, optional): list of strings that will be matched to.
                if None, existing cache is used as target strings
            top_n (int, optional): maximum number of results to populate. if None, equals length
                of tgt_texts
            scores_normalizer (str, optional): normalizer type to normalize scores. Allowed values
                are: "min_max_scaler", "standard_scaler"
            similarity_function (function, optional): if None, defaults to `pytorch_cos_sim`. If
                specified, must take two numpy-array/pytorch-tensor arguments for similarity
                computation with an optional argument to return results as numpy or tensor
            _return_as_dict (bool, optional): if the results should be returned as a dictionary of
                target_text name as keys and scores as corresponding values
            _no_sort (bool, optional): If True, results are returned without sorting. This is
                helpful at times when you wish to do additional wrapper operations on top of raw
                results and would like to save computational time without sorting.
        Returns:
            Union[dict, list[tuple]]: if _return_as_dict, returns a dictionary of tgt_texts and
                their scores, else a list of tuple each consisting of a src_text paired with its
                similarity scores with all tgt_texts as a np array (sorted list in descending order)
        """

        is_single = False
        if isinstance(src_texts, str):
            is_single = True
            src_texts = [src_texts]

        tgt_texts = [*self.cache.data.keys()] if not tgt_texts else tgt_texts
        if not tgt_texts:
            msg = (
                "The list of target texts are empty to compute similarities with the source "
                "text(s). This can happen if the embedder cache is empty due to an unloaded "
                "index or if passing in an empty list of target texts to find similarity with."
            )
            raise ValueError(msg)
        top_n = len(tgt_texts) if not top_n else top_n
        similarity_function = similarity_function or self.pytorch_cos_sim

        src_vecs = np.asarray(self.get_encodings(list(src_texts), add_to_cache=False))
        tgt_vecs = np.asarray(self.get_encodings(list(tgt_texts), add_to_cache=False))

        similarity_scores_2d = similarity_function(src_vecs, tgt_vecs)

        results = []
        for similarity_scores in similarity_scores_2d:
            similarity_scores = similarity_scores.reshape(-1)
            # Rounding sometimes helps to bring correct answers on to the list of top scored results
            similarity_scores = np.around(similarity_scores, decimals=2)

            if scores_normalizer:
                if scores_normalizer == "min_max_scaler":
                    _min = np.min(similarity_scores)
                    _max = np.max(similarity_scores)
                    denominator = (_max - _min) if (_max - _min) != 0 else 1.0
                    similarity_scores = (similarity_scores - _min) / denominator
                elif scores_normalizer == "standard_scaler":
                    _mean = np.mean(similarity_scores)
                    _std = np.std(similarity_scores)
                    denominator = _std if _std else 1.0
                    similarity_scores = (similarity_scores - _mean) / denominator
                else:
                    msg = (
                        f"Allowed values for `scores_normalizer` are only "
                        f"{['min_max_scaler', 'standard_scaler']}. Continuing without "
                        f"normalizing similarity scores."
                    )
                    logger.error(msg)

            if _return_as_dict:
                results.append(dict(zip(tgt_texts, similarity_scores)))
            else:
                if not _no_sort:  # sort results in descending scores
                    n_scores = len(similarity_scores)
                    if n_scores > top_n:
                        top_inds = similarity_scores.argpartition(n_scores - top_n)[-top_n:]
                        result = sorted(
                            [(tgt_texts[ii], similarity_scores[ii]) for ii in top_inds],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    else:
                        result = sorted(
                            zip(tgt_texts, similarity_scores),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    results.append(result)
                else:
                    result = list(zip(tgt_texts, similarity_scores))
                    results.append(result)

        if is_single:
            return results[0]

        return results

    @staticmethod
    def pytorch_cos_sim(src_vecs, tgt_vecs, return_tensor=False):
        """Computes the cosine similarity for 2d matrices

        Args:
            src_vecs: a 2d numpy array or pytorch tensor
            tgt_vecs: a 2d numpy array or pytorch tensor
            return_tensor: If False, this method returns the cosine similarity as a numpy 2d array
                instead of tensor, else returns 2d tensor output
        """

        src_vecs = torch_op("as_tensor", src_vecs)
        tgt_vecs = torch_op("as_tensor", tgt_vecs)

        if len(src_vecs.shape) == 1:
            src_vecs = src_vecs.view(1, -1)

        if len(tgt_vecs.shape) == 1:
            tgt_vecs = tgt_vecs.view(1, -1)

        if len(src_vecs.shape) != 2 or len(tgt_vecs.shape) != 2:
            msg = "Only 2-dimensional arrays/tensors are allowed in Embedder.pytorch_cos_sim()"
            raise ValueError(msg)

        # method specific to 2d tensors
        # [n_src, emb_dim] * [n_tgt, emb_dim] -> [n_src, n_tgt]
        a_norm = torch_op("normalize", src_vecs, sub="nn.functional", p=2, dim=1)
        b_norm = torch_op("normalize", tgt_vecs, sub="nn.functional", p=2, dim=1)
        similarity_scores = torch_op("mm", a_norm, b_norm.transpose(0, 1))

        if not return_tensor:
            return similarity_scores.numpy()

        return similarity_scores

    @staticmethod
    def get_hashid(**kwargs):
        string = json.dumps(kwargs, sort_keys=True)
        return Hasher(algorithm="sha256").hash(string=string)

    # deprecated method, same functionality as 'dump_cache' method
    def dump(self, cache_path=None):
        msg = (
            f"DeprecationWarning: Use {self.__class__.__name__}.dump_cache() instead of "
            f"{self.__class__.__name__}.dump()"
        )
        warnings.warn(msg, DeprecationWarning)
        self.dump_cache(cache_path=cache_path)


EMBEDDER_MAP: Dict[str, Type[Embedder]] = {}


def create_embedder_model(app_path: str, config: Dict[str, Any]) -> Embedder:
    """Creates and loads an embedder model

    Args:
        config (dict): Model settings passed in as a dictionary with
            'embedder_type' being a required key

    Returns:
        Embedder: An instance of appropriate embedder class

    Raises:
        ValueError: When model configuration is invalid or required key is missing
    """

    if "model_settings" in config and config["model_settings"]:
        # when config = {"model_settings": {"embedder_type": ..., "..": ...}}
        embedder_config = config["model_settings"]
    else:
        # when config = {"embedder_type": ..., "..": ...}}
        embedder_config = config

    embedder_type = embedder_config.get("embedder_type")
    if not embedder_type:
        raise KeyError(
            "Missing required argument in config supplied to create embedder model: 'embedder_type'"
        )

    try:
        # cache_path for embedder, if required, needs to be included as a key in the embedder_config
        return EMBEDDER_MAP[embedder_type](app_path=app_path, **embedder_config)
    except KeyError as e:
        msg = "Invalid model configuration: Unknown embedder type {!r}"
        raise ValueError(msg.format(embedder_type)) from e


def register_embedder(embedder_type: str, embedder: Type[Embedder]) -> None:
    if embedder_type in EMBEDDER_MAP:
        msg = "Embedder of type {!r} is already registered.".format(embedder_type)
        raise ValueError(msg)

    EMBEDDER_MAP[embedder_type] = embedder
