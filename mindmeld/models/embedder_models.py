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
import logging
import os
import warnings

import numpy as np
from tqdm.autonotebook import trange

from ._util import (
    _is_module_available,
    _get_module_or_attr as _getattr,
    torch_op,
)
from .embedder_base import Embedder, register_embedder
from .taggers.embeddings import WordSequenceEmbedding
from ..core import Bunch
from ..text_preparation.text_preparation_pipeline import (
    TextPreparationPipelineFactory,
)

logger = logging.getLogger(__name__)


class BertEmbedder(Embedder):  # pylint: disable=too-many-instance-attributes
    """
    Encoder class for bert models based on https://github.com/UKPLab/sentence-transformers
    """

    # Class variable to cache bert models: since pretrained transformer models like BERT are
    # generally large in size, it is optimal memory-wise if we do not load one model for each
    # object of this class. This optimization is meaningful only if BERT-like models are used for
    # inference only and not for fine-tuning
    CACHE_MODELS = {}

    def __init__(
        self,
        app_path=None,
        cache_path=None,
        pretrained_name_or_abspath=None,
        **kwargs,
    ):
        """
        Initializes a BERT based embedder from Huggingface

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.
            pretrained_name_or_abspath (str): name of the BERT model from huggingface models
                repository; arg to be used instead of deprecated arg model_name
            model_name (str, deprecated): name of the BERT model from huggingface models

            Optional keyword args that uniquely identify the embeddings of the model:
                bert_output_type (str): the output of BERT model to use, choices- 'mean', 'cls'
                quantize_model (str): if True, the BERT model is quantized
                concat_last_n_layers (int): num of hidden outputs to concat starting from last layer
                normalize_token_embs (bool): if the (sub-token) embs are to be normalized

            Optional keyword args that are required for run-time:
                device (str): Which torch.device to use for the computation
                batch_size (int): the batch size used for the computation
                output_value (str): Default sentence_embedding, to get sentence embeddings.
                    Can be set to token_embeddings to get wordpiece token embeddings.
                    Choices are `sentence_embedding` and `token_embedding`
                convert_to_numpy (bool): If true, the output is a list of numpy vectors. Else, it
                    is a list of pytorch tensors.
                convert_to_tensor (bool): If true, you get one large tensor as return. Overwrites
                    any setting from convert_to_numpy
        """

        # required libraries check
        if not _is_module_available("sentence_transformers") or not _is_module_available("torch"):
            raise ImportError(
                "Must install the extra [bert] by running `pip install mindmeld[bert]` "
                "to use the built in bert embedder."
            )

        # deprecated configs keys
        model_name = kwargs.get("model_name")
        if model_name:
            msg = (
                "The argument 'model_name' is deprecated and will be removed in future "
                "versions. Consider replacing it with 'pretrained_name_or_abspath'"
            )
            warnings.warn(msg, DeprecationWarning)
            if pretrained_name_or_abspath:
                msg = (
                    f"Must pass-in only one of 'pretrained_name_or_abspath' and 'model_name' "
                    f"params while instantiating a {self.__class__.__name__} class."
                )
                raise ValueError(msg)
            pretrained_name_or_abspath = model_name

        # configs that uniquely identify the model, used in model_id
        self.pretrained_name_or_abspath = pretrained_name_or_abspath
        if not self.pretrained_name_or_abspath:
            msg = (
                f"A valid 'pretrained_name_or_abspath' param must be passed "
                f"to instantiate {self.__class__.__name__}."
            )
            raise ValueError(msg)
        self.bert_output_type = kwargs.get("bert_output_type", "mean")
        self.quantize_model = kwargs.get("quantize_model", False)
        self.concat_last_n_layers = kwargs.get("concat_last_n_layers", 1)
        self.normalize_token_embs = kwargs.get("normalize_token_embs", False)

        # runtime configs for the embedder model
        self.device = kwargs.get(
            "device", "cuda" if torch_op("is_available", sub="cuda") else "cpu"
        )
        self._batch_size = kwargs.get("batch_size", 8)
        self._output_value = kwargs.get("output_value", "sentence_embedding")
        self._convert_to_numpy = kwargs.get("convert_to_numpy", True)
        self._convert_to_tensor = kwargs.get("convert_to_tensor", False)
        self._show_progress_bar = (
            logger.getEffectiveLevel() == logging.INFO
            or logger.getEffectiveLevel() == logging.DEBUG
        )

        # unique id for the embedder model based on specified configurations
        self._model_id = str(
            self.get_hashid(
                pretrained_name_or_abspath=self.pretrained_name_or_abspath,
                bert_output_type=self.bert_output_type,
                quantize_model=self.quantize_model,
                concat_last_n_layers=self.concat_last_n_layers,
                normalize_token_embs=self.normalize_token_embs,
            )
        )

        super().__init__(app_path=app_path, cache_path=cache_path, **kwargs)

    @staticmethod
    def _batch_to_device(batch, target_device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        tensor = _getattr("torch", "Tensor")
        for key in batch:
            if isinstance(batch[key], tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    @staticmethod
    def _num_layers(model):
        """
        Finds the number of layers in a given transformers model
        """

        if hasattr(model, "n_layers"):  # eg. xlm
            num_layers = model.n_layers
        elif hasattr(model, "layer"):  # eg. xlnet
            num_layers = len(model.layer)
        elif hasattr(model, "encoder"):  # eg. bert
            num_layers = len(model.encoder.layer)
        elif hasattr(model, "transformer"):  # eg. sentence_transformers models
            num_layers = len(model.transformer.layer)
        else:
            raise ValueError(f"Not supported model {model} to obtain number of layers")

        return num_layers

    @staticmethod
    def _get_sentence_transformers_encoder(
        name_or_path, output_type="mean", quantize=True, return_components=False
    ):
        """
        Retrieves a sentence-transformer model and returns it along with its transformer and
        pooling components.

        Args:
            name_or_path: name or path to load a huggingface model
            output_type: type of pooling required
            quantize: if the model needs to be qunatized or not
            return_components: if True, returns the Transformer and Poooling components of the
                                sentence-bert model in a Bunch data type,
                                else just returns the sentence-bert model

        Returns:
            Union[
                sentence_transformers.SentenceTransformer,
                Bunch(sentence_transformers.Transformer,
                      sentence_transformers.Pooling,
                      sentence_transformers.SentenceTransformer)
            ]
        """

        strans_models = _getattr("sentence_transformers.models")
        strans = _getattr("sentence_transformers", "SentenceTransformer")

        transformer_model = strans_models.Transformer(
            name_or_path, model_args={"output_hidden_states": True}
        )
        pooling_model = strans_models.Pooling(
            transformer_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=output_type == "cls",
            pooling_mode_max_tokens=False,
            pooling_mode_mean_tokens=output_type == "mean",
            pooling_mode_mean_sqrt_len_tokens=False,
        )
        sbert_model = strans(modules=[transformer_model, pooling_model])

        if quantize:
            if not _is_module_available("torch"):
                raise ImportError("`torch` library required to quantize models") from None

            torch_qint8 = _getattr("torch", "qint8")
            torch_nn_linear = _getattr("torch.nn", "Linear")
            torch_quantize_dynamic = _getattr("torch.quantization", "quantize_dynamic")

            transformer_model = (
                torch_quantize_dynamic(transformer_model, {torch_nn_linear}, dtype=torch_qint8)
                if transformer_model
                else None
            )
            pooling_model = (
                torch_quantize_dynamic(pooling_model, {torch_nn_linear}, dtype=torch_qint8)
                if pooling_model
                else None
            )
            sbert_model = (
                torch_quantize_dynamic(sbert_model, {torch_nn_linear}, dtype=torch_qint8)
                if sbert_model
                else None
            )

        if return_components:
            return Bunch(
                transformer_model=transformer_model,
                pooling_model=pooling_model,
                sbert_model=sbert_model,
            )

        return sbert_model

    def _encode_local(
        self,
        sentences,
        batch_size,
        show_progress_bar,
        output_value,
        convert_to_numpy,
        convert_to_tensor,
        device,
        concat_last_n_layers,
        normalize_token_embs,
    ):
        """
        Computes sentence embeddings (Note: Method largely derived from Sentence Transformers
            library to improve flexibility in encoding and pooling. Notably, `is_pretokenized` and
            `num_workers` are ignored due to deprecation in their library, retrieved 23-Feb-2021)
        """

        self.transformer_model = self.model.transformer_model
        self.pooling_model = self.model.pooling_model

        if concat_last_n_layers != 1:
            assert 1 <= concat_last_n_layers <= self._num_layers(self.transformer_model.auto_model)

        self.transformer_model.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        input_is_string = isinstance(sentences, str)
        if input_is_string:  # Cast an individual sentence to a list with length 1
            sentences = [sentences]

        self.transformer_model.to(device)
        self.pooling_model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc="Batches",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.transformer_model.tokenize(sentences_batch)
            features = self._batch_to_device(features, device)

            with torch_op("no_grad"):
                out_features_transformer = self.transformer_model.forward(features)
                token_embeddings = out_features_transformer["token_embeddings"]
                if concat_last_n_layers > 1:
                    _all_layer_embs = out_features_transformer["all_layer_embeddings"]
                    token_embeddings = torch_op(
                        "cat", _all_layer_embs[-concat_last_n_layers:], dim=-1
                    )
                if normalize_token_embs:
                    _norm_token_embeddings = torch_op(
                        "norm",
                        token_embeddings,
                        sub="linalg",
                        dim=2,
                        keepdim=True,
                    )
                    token_embeddings = token_embeddings.div(_norm_token_embeddings)
                out_features_transformer.update({"token_embeddings": token_embeddings})
                out_features = self.pooling_model.forward(out_features_transformer)

                embeddings = out_features[output_value]

                if output_value == "token_embeddings":
                    # Set token embeddings to 0 for padding tokens
                    input_mask = out_features["attention_mask"]
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch_op("stack", all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def load(self):
        model = BertEmbedder.CACHE_MODELS.get(self._model_id, None)

        if not model:
            info_msg = ""
            for name in [
                self.pretrained_name_or_abspath,
                f"sentence-transformers/{self.pretrained_name_or_abspath}",
            ]:
                try:
                    model = self._get_sentence_transformers_encoder(
                        name,
                        output_type=self.bert_output_type,
                        quantize=self.quantize_model,
                        return_components=True,
                    )
                    info_msg += (
                        f"Successfully initialized name/path `{name}` directly through "
                        f"huggingface-transformers. "
                    )
                except OSError:
                    info_msg += (
                        f"Could not initialize name/path `{name}` directly through "
                        f"huggingface-transformers. "
                    )

                if model:
                    break

            logger.info(info_msg)

            if not model:
                msg = (
                    f"Could not resolve the name/path `{self.pretrained_name_or_abspath}`. "
                    f"Please check the model name and retry."
                )
                raise Exception(msg)

            BertEmbedder.CACHE_MODELS.update({self._model_id: model})

        return model

    def encode(self, phrases):
        """Encodes input text(s) into embeddings, one vector for each phrase

        Args:
            phrases (str, list[str]): textual inputs that are to be encoded using sentence \
                                        transformers' model


        Returns:
            (Union[List[Tensor], ndarray, Tensor]): By default, a numpy array is returned.
                If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy
                matrix is returned.
        """

        if not phrases:
            return []

        show_progress_bar = (
            self._show_progress_bar and (len(phrases) if isinstance(phrases, list) else 1) > 1
        )

        # `False` for first call but might not for the subsequent calls
        _use_sbert_model = getattr(self, "_use_sbert_model", False)

        results = None
        if not _use_sbert_model:
            try:
                # this snippet is to reduce dependency on sentence-transformers library
                #   note that currently, the dependency is not fully eliminated due to backwards
                #   compatibility issues in huggingface-transformers between older (python 3.6)
                #   and newer (python >=3.7) versions which needs more conditions to be implemented
                #   in `_encode_local` and hence will be addressed in future work
                # TODO: eliminate depedency on sentence-transformers library
                results = self._encode_local(
                    phrases,
                    batch_size=self._batch_size,
                    show_progress_bar=show_progress_bar,
                    output_value=self._output_value,
                    convert_to_numpy=self._convert_to_numpy,
                    convert_to_tensor=self._convert_to_tensor,
                    device=self.device,
                    concat_last_n_layers=self.concat_last_n_layers,
                    normalize_token_embs=self.normalize_token_embs,
                )
                setattr(self, "_use_sbert_model", False)
            except TypeError as e:
                logger.error(e)
                if self.concat_last_n_layers != 1 or self.normalize_token_embs:
                    msg = (
                        f"{'concat_last_n_layers,' if self.concat_last_n_layers != 1 else ''} "
                        f"{'normalize_token_embs' if self.normalize_token_embs else ''} "
                        f"ignored as resorting to using encode methods from sentence-transformers"
                    )
                    logger.warning(msg)
                setattr(self, "_use_sbert_model", True)

        if getattr(self, "_use_sbert_model"):
            results = self.model.sbert_model.encode(
                phrases,
                batch_size=self._batch_size,
                show_progress_bar=show_progress_bar,
                output_value=self._output_value,
                convert_to_numpy=self._convert_to_numpy,
                convert_to_tensor=self._convert_to_tensor,
                device=self.device,
            )

        return results

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs"""
        return self._model_id


class GloveEmbedder(Embedder):
    """
    Encoder class for GloVe embeddings as described here: https://nlp.stanford.edu/projects/glove/
    """

    DEFAULT_EMBEDDING_DIM = 300

    def __init__(self, app_path=None, cache_path=None, **kwargs):
        """
        Initializes a GloVe embedder.

        Args:
            app_path (str): Path of the app used to create cache folder to dump encodings
            cache_path (str): A .pkl path where the embeddings are to be cached. If provided,
                discards the app_path information.

            Optional keyword args that uniquely identify the embeddings of the model:
                token_embedding_dimension (str): The token dimension of GloVe embedder to load
                token_pretrained_embedding_filepath (str): The path where GloVe embeddings are
                    available. If its None, an appropriate file will be downloaded to
                    mindmeld/data/ folder and used.
        """

        self.token_embedding_dimension = kwargs.get(
            "token_embedding_dimension", self.DEFAULT_EMBEDDING_DIM
        )
        self.token_pretrained_embedding_filepath = kwargs.get("token_pretrained_embedding_filepath")

        # Create a custom pipeline config as the default config for en language eliminates some
        # punctuations that can be required in tasks such as entity resolution.
        pipeline_config = {
            "language": "en",
            "tokenizer": "WhiteSpaceTokenizer",
            "preprocessors": [],
            "normalizers": [],
            "stemmer": None,
            "keep_special_chars": True,
        }
        self.text_preparation_pipeline = (
            TextPreparationPipelineFactory.create_text_preparation_pipeline(**pipeline_config)
        )

        # unique id for the embedder model based on specified configurations
        self._model_id = str(
            self.get_hashid(
                token_embedding_dimension=self.token_embedding_dimension,
                token_pretrained_embedding_filepath=os.path.abspath(
                    self.token_pretrained_embedding_filepath
                )
                if self.token_pretrained_embedding_filepath
                else "default",
            )
        )

        super().__init__(app_path=app_path, cache_path=cache_path, **kwargs)

    def load(self):
        return WordSequenceEmbedding(
            0,
            self.token_embedding_dimension,
            self.token_pretrained_embedding_filepath,
            use_padding=False,
        )

    def encode(self, text_list):
        token_list = [self._tokenize(text) for text in text_list]
        vector_list = [self.model.encode_sequence_of_tokens(tl) for tl in token_list]
        encoded_vecs = []
        for vl in vector_list:
            if len(vl) == 1:
                encoded_vecs.append(vl[0])
            else:
                encoded_vecs.append(np.average(vl, axis=0))
        return encoded_vecs

    def _tokenize(self, text):
        return [t["entity"] for t in self.text_preparation_pipeline.tokenize_and_normalize(text)]

    def dump(self, cache_path=None):
        """Dumps the cache to disk."""
        super().dump(cache_path=cache_path)
        self.model.save_embeddings()

    @property
    def model_id(self):
        """Returns a unique hash representation of the embedder model based on its name and configs"""
        return self._model_id


if _is_module_available("sentence_transformers"):
    register_embedder("bert", BertEmbedder)

register_embedder("glove", GloveEmbedder)
