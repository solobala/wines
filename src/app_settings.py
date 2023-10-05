import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()


def json_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    """
    A simple settings source that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """
    encoding = settings.__config__.env_file_encoding
    p = os.path.abspath('config.json ')
    try:
        return json.loads(Path('src/config.json').read_text(encoding))
    except FileNotFoundError:
        # return json.loads(Path('config.json').read_text(encoding))
        return json.loads(Path(p).read_text(encoding))


class AppSettings(BaseSettings):
    """ Settings for vine project.
    """
    TYPE_MODEL_FILE: str
    TYPE_PREDICTION_FILE: str
    DATA_FOLDER: str
    INTERIM_DATA_FILE: str
    INTERIM_DATA_FOLDER: str
    CNN_MODEL_FILE: str
    TRANSFORM_MODEL_FILE: str
    QUALITY_PREDICTION_FILE: str
    PREDICTION_FOLDER: str
    PROCESSED_TYPE_DATA_FILE: str
    PROCESSED_QUALITY_DATA_FILE: str
    PROCESSED_QUALITY_TRANSFORMED_DATA_FILE: str
    PROCESSED_TYPE_DATA_FILE_TO_PREDICTION: str
    PROCESSED_QUALITY_DATA_FILE_TO_PREDICTION: str
    RAW_DATA_FILE: str
    RAW_DATA_FOLDER: str
    RAW_DATA_FILE_TO_PREDICTION: str
    TYPE_METRICS: str
    CNN_METRICS: str
    TRANSFORM_METRICS: str
    KAGGLE_CONFIG_DIR: str

    class Config:
        env_file_encoding = 'utf-8'

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )
