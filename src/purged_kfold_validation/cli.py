"""Distribution-native governed arbitrary-feature upload CLI."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
from math import comb
from pathlib import Path
import sys
from typing import TYPE_CHECKING, NoReturn

import numpy as np

from .domain import ModelSpec
from .effectiveness import (
    EffectivenessComparisonConfig,
    run_cpcv_effectiveness_comparison,
)
from .errors import AdapterValidationError, UploadLimitError, ValidationError

if TYPE_CHECKING:
    from .adapters.upload import FeatureUploadLimits, LoadedFeatureUpload


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        command = sys.argv[1] if len(sys.argv) > 1 else "arguments"
        stage = command if command in {"audit", "evaluate"} else "arguments"
        _print_json(
            {
                "error_type": "ArgumentError",
                "message": "command arguments are invalid",
                "stage": stage,
                "status": "rejected",
            }
        )
        raise SystemExit(2)


class RidgeEstimator:
    """Deterministic fold-local standardized ridge baseline."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeEstimator:
        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale = np.where(scale == 0.0, 1.0, scale)
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        assert self.scale is not None
        assert self.coef is not None
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        return np.asarray(design @ self.coef)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--max-file-bytes", type=_positive_integer, default=536_870_912)
    parser.add_argument("--max-rows", type=_positive_integer, default=1_000_000)
    parser.add_argument("--max-features", type=_positive_integer, default=512)
    parser.add_argument("--max-combinations", type=_positive_integer, default=10_000)
    parser.add_argument("--max-columns", type=_positive_integer, default=2_048)
    parser.add_argument(
        "--max-uncompressed-bytes", type=_positive_integer, default=2_147_483_648
    )


def parse_args() -> argparse.Namespace:
    parser = JsonArgumentParser(description="Governed arbitrary-feature upload")
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=JsonArgumentParser
    )
    audit = commands.add_parser("audit")
    _common_arguments(audit)
    evaluate = commands.add_parser("evaluate")
    _common_arguments(evaluate)
    evaluate.add_argument("--n-groups", type=int, default=6)
    evaluate.add_argument("--n-test-groups", type=int, default=2)
    evaluate.add_argument("--walk-forward-splits", type=int, default=5)
    evaluate.add_argument("--embargo-sessions", type=int, default=20)
    evaluate.add_argument("--pre-test-gap-sessions", type=int, default=20)
    evaluate.add_argument("--min-train-observations", type=int, default=10_000)
    evaluate.add_argument("--min-train-sessions", type=int, default=252)
    evaluate.add_argument("--min-train-assets", type=int, default=20)
    evaluate.add_argument("--ridge-alpha", type=float, default=1.0)
    schema = commands.add_parser("schema")
    schema.add_argument("--kind", choices=("manifest", "mapping"), required=True)
    example = commands.add_parser("example")
    example.add_argument(
        "--name",
        choices=("raw", "stationary", "intentional-leak"),
        required=True,
    )
    example.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _limits(args: argparse.Namespace) -> FeatureUploadLimits:
    from .adapters.upload import FeatureUploadLimits

    return FeatureUploadLimits(
        max_file_bytes=args.max_file_bytes,
        max_rows=args.max_rows,
        max_features=args.max_features,
        max_combinations=args.max_combinations,
        max_columns=args.max_columns,
        max_uncompressed_bytes=args.max_uncompressed_bytes,
    )


def _audit_payload(
    loaded: LoadedFeatureUpload,
    *,
    limits: FeatureUploadLimits,
    stage: str,
) -> dict[str, object]:
    dataset = loaded.governed.dataset
    return {
        "status": "success",
        "stage": stage,
        "input": {
            "file_name": loaded.file_name,
            "file_bytes": loaded.file_bytes,
        },
        "limits": limits.canonical(),
        "dataset": {
            "observations": len(dataset.sample_ids),
            "features": int(dataset.features.shape[1]),
            "sessions": len(set(dataset.sessions)),
            "assets": len(set(dataset.asset_ids or ())),
            "digest": dataset.digest,
            "feature_manifest_digest": dataset.feature_manifest_digest,
        },
        "governance": loaded.governed.receipt.canonical(),
    }


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main() -> int:
    args = parse_args()
    stage = str(args.command)
    try:
        if stage == "schema":
            schema_names = {
                "manifest": "feature-manifest.schema.json",
                "mapping": "upload-mapping.schema.json",
            }
            resource = (
                files("purged_kfold_validation")
                .joinpath("resources", "feature_upload")
                .joinpath(schema_names[args.kind])
            )
            _print_json(json.loads(resource.read_text(encoding="utf-8")))
            return 0
        if stage == "example":
            file_names = ("features.csv", "manifest.json", "mapping.json")
            package_directory = (
                files("purged_kfold_validation")
                .joinpath("resources", "feature_upload", "examples")
                .joinpath(args.name)
            )
            content = {
                name: package_directory.joinpath(name).read_bytes()
                for name in file_names
            }
            output_directory = Path(args.output_dir)
            targets = tuple(output_directory / name for name in file_names)
            if output_directory.exists() and not output_directory.is_dir():
                raise AdapterValidationError(
                    "example output directory must be a directory"
                )
            if any(target.exists() for target in targets):
                raise AdapterValidationError(
                    "example export refuses to overwrite existing files"
                )
            output_directory.mkdir(parents=True, exist_ok=True)
            for target in targets:
                with target.open("xb") as handle:
                    handle.write(content[target.name])
            _print_json(
                {
                    "example": args.name,
                    "files": list(file_names),
                    "stage": "example",
                    "status": "success",
                }
            )
            return 0
        from .adapters.upload import load_governed_feature_upload

        limits = _limits(args)
        loaded = load_governed_feature_upload(
            Path(args.data),
            manifest_path=Path(args.manifest),
            mapping_path=Path(args.mapping),
            limits=limits,
        )
        payload = _audit_payload(loaded, limits=limits, stage=stage)
        if stage == "evaluate":
            combination_count = comb(args.n_groups, args.n_test_groups)
            if combination_count > limits.max_combinations:
                raise UploadLimitError(
                    "CPCV combination count exceeds max_combinations"
                )
            dataset = loaded.governed.dataset
            model = ModelSpec(
                name="fold-local-ridge",
                version="1-upload-governed",
                parameters={"alpha": args.ridge_alpha},
            )
            comparison = run_cpcv_effectiveness_comparison(
                dataset,
                label_roll_clean=np.ones(len(dataset.sample_ids), dtype=bool),
                estimator_factory=lambda: RidgeEstimator(args.ridge_alpha),
                model_spec=model,
                config=EffectivenessComparisonConfig(
                    n_groups=args.n_groups,
                    n_test_groups=args.n_test_groups,
                    walk_forward_splits=args.walk_forward_splits,
                    embargo_sessions=args.embargo_sessions,
                    pre_test_gap_sessions=args.pre_test_gap_sessions,
                    min_train_observations=args.min_train_observations,
                    min_train_sessions=args.min_train_sessions,
                    min_train_assets=args.min_train_assets,
                    max_combinations=limits.max_combinations,
                ),
            )
            payload["evaluation"] = {
                "estimator": "fold-local-ridge",
                "cpcv_combinations": combination_count,
            }
            payload["report"] = comparison.canonical()
        _print_json(payload)
        return 0
    except ImportError:
        _print_json(
            {
                "error_type": "OptionalDependencyError",
                "message": (
                    "audit/evaluate require the upload extra: "
                    "pip install purged-kfold-validation[upload]"
                ),
                "stage": stage,
                "status": "rejected",
            }
        )
        return 2
    except ValidationError as exc:
        _print_json(
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stage": stage,
                "status": "rejected",
            }
        )
        return 2
    except (OSError, ValueError, TypeError):
        _print_json(
            {
                "error_type": "UploadParseError",
                "message": "upload could not be parsed",
                "stage": stage,
                "status": "rejected",
            }
        )
        return 2
