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

"""A module containing various utility functions for MindMeld.
These are capabilities that do not have an obvious home within the existing
project structure.
"""
import logging
import os
import sys
from email.utils import parsedate

import py

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "mindmeld.cfg"

def configure_logs(**kwargs):
    """Helper method for easily configuring logs from the python shell.

    Args:
        level (TYPE, optional): A logging level recognized by python's logging module.
    """
    level = kwargs.get("level", logging.INFO)
    log_format = kwargs.get("format", "%(message)s")
    logging.basicConfig(stream=sys.stdout, format=log_format)
    package_logger = logging.getLogger(__package__)
    package_logger.setLevel(level)


def load_configuration():
    """Loads a configuration file (mindmeld.cfg) for the current app. The
    file is located by searching first in the current directory, and in parent
    directories.
    """
    config_file = _find_config_file()
    if config_file:
        logger.debug("Using config file at '%s'", config_file)
        # Do the thing
        iniconfig = py.iniconfig.IniConfig(config_file)  # pylint: disable=no-member
        config = {}
        config["app_name"] = iniconfig.get("mindmeld", "app_name")
        config["app_path"] = iniconfig.get("mindmeld", "app_path")
        config["use_quarry"] = iniconfig.get("mindmeld", "use_quarry")
        config["input_method"] = iniconfig.get("mindmeld", "input_method")
        # resolve path if necessary
        if config["app_path"] and not os.path.isabs(config["app_path"]):
            config_dir = os.path.dirname(config_file)
            config["app_path"] = os.path.abspath(
                os.path.join(config_dir, config["app_path"])
            )
        return config
    else:
        logger.debug("No config file was found.")


def _find_config_file():
    prev_dir = None
    current_dir = os.getcwd()

    while prev_dir != current_dir:
        config_file = os.path.join(current_dir, CONFIG_FILE_NAME)
        if os.path.isfile(config_file):
            # found a config file!
            return config_file

        # go up one directory
        prev_dir = current_dir
        current_dir = os.path.abspath(os.path.join(current_dir, ".."))

    return None


def get_pattern(rule):
    """Convert a rule represented as a dictionary with the keys "domains", "intents",
    "files" into a regex pattern.

    Args:
        rule (dict): An annotation or augmentation rule.

    Returns:
        pattern (str): Regex pattern specifying allowed file paths.
    """
    pattern = [rule[x] for x in ["domains", "intents", "files"]]
    return ".*/" + "/".join(pattern)


def read_path_queries(filepath):
    """Reads queries from given file path.

        Args:
            filepath (str): File path to read from.

        Returns:
            queries (list): List of queries.
    """
    with open(filepath, "r") as f:
        queries = f.readlines()
    return queries


def write_to_file(filepath, queries, suffix):
    """Writes queries to a new file in the path with given suffix.

    Args:
        filepath (str): File path to the original file.
        queries (list): List of queries to be written to file.
    """
    write_path = filepath.rstrip(".txt") + suffix

    with open(write_path, "w") as outfile:
        for query in queries:
            outfile.write(query.rstrip() + "\n")
