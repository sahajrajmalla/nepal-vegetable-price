"""
utils.py — Shared Utilities for the Kalimati Price Forecasting Pipeline
========================================================================

Provides:
    • YAML configuration loading & validation
    • Reproducible seed setting (NumPy, PyTorch, Python)
    • Project-wide logger factory
    • Directory creation helpers
    • Timer context manager for profiling

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # kalimati_price_forecasting/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config. Defaults to ``configs/default.yaml``.

    Returns
    -------
    dict
        Nested configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    cfg = _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve relative path values in the config against PROJECT_ROOT."""
    path_keys = {
        ("data", "raw_dir"),
        ("data", "cleaned_dir"),
        ("data", "interim_dir"),
        ("output", "figures_dir"),
        ("output", "models_dir"),
        ("output", "reports_dir"),
        ("logging", "log_file"),
    }
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if val and not os.path.isabs(val):
                cfg[section][key] = str(PROJECT_ROOT / val)
    return cfg


def get_config_value(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely traverse nested config dict.

    Example
    -------
    >>> get_config_value(cfg, "models", "ml", "xgboost", "enabled")
    True
    """
    node = cfg
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy,
    and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(
    name: str = "kalimati",
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Create and configure a named logger with console and optional file handlers.

    Only the root ``kalimati`` logger receives handlers.  Child loggers
    (e.g. ``kalimati.preprocessing``) simply propagate to the root,
    preventing duplicate log lines.

    Parameters
    ----------
    name : str
        Logger name (used in log output prefix).
    log_file : str, optional
        Path to log file. If None, file logging is disabled.
    level : str
        Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    console : bool
        Whether to attach a console (stdout) handler.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Only attach handlers to the root project logger to avoid duplicates
    is_root = (name == "kalimati")

    if is_root:
        logger.handlers.clear()
        logger.propagate = False  # Don't propagate to Python root logger

        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    else:
        # Child loggers: just set level, propagate to parent
        logger.handlers.clear()
        logger.propagate = True

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """
    Create all output directories defined in the configuration.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration dictionary.
    """
    dirs_to_create = [
        cfg.get("data", {}).get("cleaned_dir"),
        cfg.get("data", {}).get("interim_dir"),
        cfg.get("output", {}).get("figures_dir"),
        cfg.get("output", {}).get("models_dir"),
        cfg.get("output", {}).get("reports_dir"),
    ]
    for d in dirs_to_create:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def timer(label: str = "Block", logger: Optional[logging.Logger] = None):
    """
    Context manager that logs elapsed wall-clock time.

    Usage
    -----
    >>> with timer("Training XGBoost", logger):
    ...     model.fit(X, y)
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    msg = f"⏱  {label} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Commodity helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_selected_commodities(cfg: Dict[str, Any]) -> List[str]:
    """
    Return the list of commodities to process based on configuration.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    list of str
        Commodity names to forecast.
    """
    return cfg.get("commodities", {}).get("selected", [])


def sanitize_commodity_name(name: str) -> str:
    """
    Convert commodity name to a filesystem-safe slug.

    Example
    -------
    >>> sanitize_commodity_name("Tomato Big(Nepali)")
    'tomato_big_nepali'
    """
    slug = name.lower()
    for ch in "()/ ":
        slug = slug.replace(ch, "_")
    slug = slug.strip("_")
    # Collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


# ─────────────────────────────────────────────────────────────────────────────
# Version logging
# ─────────────────────────────────────────────────────────────────────────────


def log_environment(logger: logging.Logger) -> None:
    """Log key library versions for reproducibility."""
    import pandas as pd
    import sklearn

    versions = {
        "Python": sys.version.split()[0],
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Scikit-learn": sklearn.__version__,
    }

    try:
        import xgboost
        versions["XGBoost"] = xgboost.__version__
    except ImportError:
        pass

    try:
        import torch
        versions["PyTorch"] = torch.__version__
    except ImportError:
        pass

    try:
        import statsmodels
        versions["Statsmodels"] = statsmodels.__version__
    except ImportError:
        pass

    try:
        import pmdarima
        versions["pmdarima"] = pmdarima.__version__
    except ImportError:
        pass

    # Git commit hash for reproducibility
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).strip().decode()
        versions["Git Commit"] = commit
    except Exception:
        pass

    logger.info("── Environment ──")
    for lib, ver in versions.items():
        logger.info(f"  {lib:15s} : {ver}")
