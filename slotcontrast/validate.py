"""Evaluate a trained checkpoint on the validation split.

This mirrors the validation pass that runs during training: it rebuilds the model from the same
config, restores the weights from a checkpoint, and runs PyTorch Lightning's standard validation
loop. The same metrics are computed and logged the same way as during training
(``val/loss`` plus whatever is configured under ``val_metrics``), to TensorBoard + CSV (and Comet
if ``comet.project`` is set), and printed to the console.

Usage:
    python -m slotcontrast.validate <config> --checkpoint <ckpt> [config_overrides...]
"""
import argparse
import logging
import os
import pathlib
import random
import warnings
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_info as log_info

from slotcontrast import configuration, data, metrics, models, utils

TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument(
    "--no-interactive", action="store_true", help="If running in non-interactive environment"
)
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default="./logs", help="Path to log directory")
parser.add_argument(
    "--checkpoint", required=True, help="Path to the .ckpt file to evaluate"
)
parser.add_argument("config", help="Configuration to run")
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")


def _setup_loggers(args, log_path: pathlib.Path, config) -> Dict[str, pl.loggers.logger.Logger]:
    if args.dry:
        return {}

    loggers = {}
    if not args.no_tensorboard:
        # Tensorboard logs go to <log_dir>/<tensorboard_subdir>/
        loggers["tensorboard"] = pl.loggers.TensorBoardLogger(
            save_dir=log_path, name=TENSORBOARD_SUBDIR, version=""
        )

    if "comet" in config and config.comet is not None and config.comet.project is not None:
        mode = "create" if config.comet.run_id is None else "get"
        loggers["comet"] = pl.loggers.CometLogger(
            project_name=config.comet.project,
            experiment_name=config.comet.run_name,
            experiment_key=config.comet.run_id,
            mode=mode,
        )
        loggers["comet"].experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    # CSV logs go to <log_dir>/<metrics_subdir>/version_N/metrics.csv
    loggers["csv"] = pl.loggers.CSVLogger(save_dir=log_path, name=METRICS_SUBDIR)

    return loggers


def main(args, config_overrides=None):
    rank_zero = utils.get_rank() == 0
    if config_overrides is None:
        config_overrides = args.config_overrides
    config = configuration.load_config(args.config, config_overrides)

    if not args.verbose or args.quiet:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        from pytorch_lightning.utilities.warnings import PossibleUserWarning

        warnings.filterwarnings("ignore", category=PossibleUserWarning)
    if args.quiet:
        warnings.filterwarnings("ignore", category=UserWarning)

    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint file {checkpoint_path} does not exist.")

    # Setup log path (separate from training logs so we never clobber them)
    log_path: Optional[pathlib.Path] = None
    if not args.dry:
        log_path = utils.make_log_dir(
            args.log_dir, f"{config.experiment_name}_eval", config.experiment_group
        )
        log_info(f"Using {log_path} as evaluation output directory")

    # Setup random seed. Set it via the config (e.g. `seed=42` on the CLI) for a reproducible run:
    # with `dataset.val_shuffle=true` a fixed seed also fixes the (otherwise random) val order.
    if config.seed is not None:
        seed = config.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    log_info(f"Using random seed {seed}.")
    config.seed = pl.seed_everything(seed, workers=True)

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    if args.verbose:
        log_info(str(dataset))

    if config.val_metrics is not None:
        val_metrics = {name: metrics.build(cfg) for name, cfg in config.val_metrics.items()}
    else:
        raise ValueError("No `val_metrics` configured; nothing to evaluate.")

    model = models.build(config.model, config.optimizer, None, val_metrics)

    loggers = _setup_loggers(args, log_path, config)

    # Save the resolved config alongside the eval logs for provenance.
    if rank_zero and log_path and not (log_path / "settings.yaml").exists():
        configuration.save_config(log_path / "settings.yaml", config)

    if "tensorboard" in loggers:
        loggers["tensorboard"].log_hyperparams(config)

    log_info(f"Configuration:\n{OmegaConf.to_yaml(config, resolve=True)}")
    log_info(f"Evaluating checkpoint {checkpoint_path}")

    trainer = pl.Trainer(
        accelerator="auto",
        # Force a single device: EpisodesDataModule has no DDP validation padding, so multi-GPU
        # would mis-aggregate metrics across ranks.
        devices=1,
        default_root_dir=log_path,
        logger=[logger for logger in loggers.values()] if loggers else False,
        enable_progress_bar=(not args.quiet and not args.no_interactive),
        enable_model_summary=not args.quiet,
        enable_checkpointing=False,
    )

    # `ckpt_path` makes Lightning restore the model weights before running validation.
    results = trainer.validate(model=model, datamodule=dataset, ckpt_path=str(checkpoint_path))

    if rank_zero:
        # Print every metric to stdout (PL's `rank_zero_info` would be suppressed at the default
        # log level). `results` is one dict of {metric_name: value} per validation dataloader.
        print("\n=== Validation metrics ===")
        for idx, result in enumerate(results):
            if len(results) > 1:
                print(f"[dataloader {idx}]")
            for name, value in sorted(result.items()):
                if isinstance(value, torch.Tensor):
                    value = value.item()
                print(f"{name}: {value}")

    return results


if __name__ == "__main__":
    main(parser.parse_args())
