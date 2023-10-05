from functools import lru_cache
from pydantic import BaseSettings, Field
from pydantic.types import PositiveInt
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv('.env'))


class _Settings(BaseSettings):
    class Config:
        """Configuration of settings."""
        #: str: env file encoding.
        env_file_encoding = "utf-8"
        #: str: allow custom fields in model.
        arbitrary_types_allowed = True


class Settings(_Settings):
    """ Settings for vine project.
    """
    USER_NAME: str = Field(" ", env='USER_NAME')
    KEY: str = Field(" ", env='KEY')
    RANDOM_SEED: PositiveInt = Field(1, env='RANDOM_SEED')


@lru_cache
def get_settings(env_file: str = ".env"):
    """Create settings instance."""
    _file = find_dotenv(env_file)
    return Settings(_env_file=_file)
