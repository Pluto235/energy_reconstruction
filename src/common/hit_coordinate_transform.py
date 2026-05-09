from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from .WCDA_configuration import WCDAConfig


@lru_cache(maxsize=1)
def get_default_wcda_config() -> WCDAConfig:
    return WCDAConfig()


def detector_ids_to_global_points(
    detector_ids: Sequence[int],
    *,
    config: Optional[WCDAConfig] = None,
) -> np.ndarray:
    ids = np.asarray(detector_ids, dtype=np.int64).reshape(-1)
    cfg = config if config is not None else get_default_wcda_config()
    x, y = cfg.get_xy(ids)
    return np.column_stack([x, y]).astype(np.float32)


def build_hit_points(
    features: np.ndarray,
    *,
    detector_ids: Optional[Sequence[int]] = None,
    coordinate_system: str = "global",
    config: Optional[WCDAConfig] = None,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(f"Expected features with shape (N, >=2), got {features.shape}")

    coordinate_system = coordinate_system.lower().strip()
    if coordinate_system == "local":
        return np.column_stack([features[:, 0], features[:, 1]]).astype(np.float32)

    if coordinate_system == "global":
        if detector_ids is None:
            raise ValueError("detector_ids is required when coordinate_system='global'")
        ids = np.asarray(detector_ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != features.shape[0]:
            raise ValueError(
                "detector_ids and features must have the same number of hits, "
                f"got {ids.shape[0]} vs {features.shape[0]}"
            )
        return detector_ids_to_global_points(ids, config=config)

    raise ValueError(f"Unsupported coordinate_system: {coordinate_system!r}")
