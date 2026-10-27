"""Loading and validation of config.yaml."""

from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel

from pipelines.common.timeutil import parse_duration


class StorageCfg(BaseModel):
    root: Path = Path("data")


class GammaCfg(BaseModel):
    base_url: str
    series_id: int
    slug_template: str
    window_hours: int = 6
    page_limit: int = 100
    settle_margin_s: int = 600
    rate_per_s: float = 4
    timeout_s: float = 30
    max_retries: int = 6


class DataApiCfg(BaseModel):
    base_url: str
    page_limit: int = 10000
    concurrency: int = 12
    rate_per_s: float = 15
    timeout_s: float = 60
    max_retries: int = 6


class BinanceCfg(BaseModel):
    base_url: str
    symbol: str
    margin_days: int = 1
    concurrency: int = 4
    timeout_s: float = 120
    max_retries: int = 6


class Config(BaseModel):
    asset: str
    duration: str
    n_markets: int
    storage: StorageCfg
    gamma: GammaCfg
    data_api: DataApiCfg
    binance: BinanceCfg

    @property
    def duration_s(self) -> int:
        return parse_duration(self.duration)

    def slug(self, start_ts: int) -> str:
        return self.gamma.slug_template.format(
            asset=self.asset, duration=self.duration, start_ts=start_ts
        )


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
