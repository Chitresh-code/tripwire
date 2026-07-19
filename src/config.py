"""Loads settings from configs/*.yaml instead of hardcoding values in code."""

from pathlib import Path
from typing import Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class FeatureSettings(BaseSettings):
    high_amount_threshold: float

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            YamlConfigSettingsSource(settings_cls, yaml_file=_CONFIGS_DIR / "features.yaml"),
        )
