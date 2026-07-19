"""Loads settings from configs/*.yaml instead of hardcoding values in code."""

from datetime import datetime
from pathlib import Path
from typing import Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class _YamlSettings(BaseSettings):
    """Base for settings loaded from a single configs/*.yaml file named after the subclass."""

    @classmethod
    def _yaml_filename(cls) -> str:
        raise NotImplementedError

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
            YamlConfigSettingsSource(settings_cls, yaml_file=_CONFIGS_DIR / cls._yaml_filename()),
        )


class FeatureSettings(_YamlSettings):
    high_amount_threshold: float
    velocity_window_minutes: float

    @classmethod
    def _yaml_filename(cls) -> str:
        return "features.yaml"


class IngestionSettings(_YamlSettings):
    paysim_epoch: datetime  # arbitrary real-looking start date; PaySim's `step` is hours since this point

    @classmethod
    def _yaml_filename(cls) -> str:
        return "ingestion.yaml"
