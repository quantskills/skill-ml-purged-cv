"""Agent-neutral one-request CLI over the governed upload engine."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import NoReturn, cast

from . import __version__


MAX_REQUEST_BYTES = 1_048_576
DEFAULT_TIMEOUT_SECONDS = 900
COMMON_OPTIONS = {
    "max_file_bytes",
    "max_rows",
    "max_features",
    "max_combinations",
    "max_columns",
    "max_uncompressed_bytes",
}
EVALUATION_OPTIONS = {
    "n_groups",
    "n_test_groups",
    "walk_forward_splits",
    "embargo_sessions",
    "pre_test_gap_sessions",
    "min_train_observations",
    "min_train_sessions",
    "min_train_assets",
    "ridge_alpha",
}
ZERO_ALLOWED_OPTIONS = {"embargo_sessions", "pre_test_gap_sessions"}
MINIMUM_TWO_OPTIONS = {"n_groups"}
REQUIRED_REQUEST_KEYS = {
    "schema_version",
    "action",
    "data_path",
    "manifest_path",
    "mapping_path",
}
OPTIONAL_REQUEST_KEYS = {"options", "timeout_seconds"}


class AgentRequestError(ValueError):
    """Raised when the one-file Agent request violates its closed contract."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_json(
            _error_envelope(
                action="arguments",
                error_type="ArgumentError",
                message="command arguments are invalid",
            )
        )
        raise SystemExit(2)


def _parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(
        description="Agent-neutral leakage-safe financial validation"
    )
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=JsonArgumentParser
    )
    run = commands.add_parser("run")
    run.add_argument("--request", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("--kind", choices=("request", "result"), required=True)
    example = commands.add_parser("example")
    example.add_argument("--output-dir", required=True)
    commands.add_parser("demo")
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _error_envelope(
    *, action: str, error_type: str, message: str, request_digest: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1",
        "status": "rejected",
        "action": action,
        "engine": {"name": "purged-kfold-validation", "version": __version__},
        "authoritative_cli_result": {},
        "warnings": [],
        "errors": [{"error_type": error_type, "message": message}],
    }
    if request_digest is not None:
        payload["request_digest"] = request_digest
    return payload


def _resource_json(name: str) -> dict[str, object]:
    resource = files("purged_kfold_validation").joinpath(
        "resources", "agent_skill", name
    )
    loaded = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"packaged resource is not an object: {name}")
    return cast(dict[str, object], loaded)


def _read_request(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise AgentRequestError("request must be an existing local JSON file")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise AgentRequestError("request exceeds the 1 MiB request limit")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentRequestError("request could not be parsed as UTF-8 JSON") from exc
    if not isinstance(loaded, dict):
        raise AgentRequestError("request root must be a JSON object")
    request = cast(dict[str, object], loaded)
    digest = sha256(
        json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    _validate_request(request)
    return request, digest


def _validate_request(request: Mapping[str, object]) -> None:
    keys = set(request)
    missing = REQUIRED_REQUEST_KEYS - keys
    unknown = keys - REQUIRED_REQUEST_KEYS - OPTIONAL_REQUEST_KEYS
    if missing or unknown:
        raise AgentRequestError(
            "request fields do not match the closed version-1 contract"
        )
    if request["schema_version"] != "1":
        raise AgentRequestError("unsupported request schema_version")
    action = request["action"]
    if action not in {"audit", "evaluate"}:
        raise AgentRequestError("action must be audit or evaluate")
    for key in ("data_path", "manifest_path", "mapping_path"):
        if not isinstance(request[key], str) or not request[key]:
            raise AgentRequestError(f"{key} must be a non-empty string")
    options = request.get("options", {})
    if not isinstance(options, dict):
        raise AgentRequestError("options must be an object")
    allowed = COMMON_OPTIONS | (EVALUATION_OPTIONS if action == "evaluate" else set())
    if set(options) - allowed:
        raise AgentRequestError(
            "options contain fields that are invalid for the action"
        )
    for name, value in options.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentRequestError(f"option {name} must be numeric")
        if name == "ridge_alpha":
            if value <= 0:
                raise AgentRequestError("option ridge_alpha must be greater than zero")
            continue
        if not isinstance(value, int):
            raise AgentRequestError(f"option {name} must be an integer")
        minimum = (
            0
            if name in ZERO_ALLOWED_OPTIONS
            else 2
            if name in MINIMUM_TWO_OPTIONS
            else 1
        )
        if value < minimum:
            raise AgentRequestError(f"option {name} must be at least {minimum}")
    timeout = request.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 3600
    ):
        raise AgentRequestError("timeout_seconds must be an integer from 1 to 3600")


def _resolve_input(request_path: Path, value: object) -> Path:
    assert isinstance(value, str)
    path = Path(value)
    if not path.is_absolute():
        path = request_path.parent / path
    return path.resolve()


def _engine_command(request_path: Path, request: Mapping[str, object]) -> list[str]:
    action = cast(str, request["action"])
    command = [
        sys.executable,
        "-m",
        "purged_kfold_validation",
        action,
        "--data",
        str(_resolve_input(request_path, request["data_path"])),
        "--manifest",
        str(_resolve_input(request_path, request["manifest_path"])),
        "--mapping",
        str(_resolve_input(request_path, request["mapping_path"])),
    ]
    options = cast(dict[str, object], request.get("options", {}))
    for name in sorted(options):
        command.extend((f"--{name.replace('_', '-')}", str(options[name])))
    return command


def _execute_request(request_path: Path) -> tuple[int, dict[str, object]]:
    try:
        request, request_digest = _read_request(request_path)
    except AgentRequestError as exc:
        return 2, _error_envelope(
            action="request",
            error_type=type(exc).__name__,
            message=str(exc),
        )
    action = cast(str, request["action"])
    command = _engine_command(request_path, request)
    timeout = cast(int, request.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 2, _error_envelope(
            action=action,
            error_type="EngineTimeoutError",
            message="validation engine exceeded the declared timeout",
            request_digest=request_digest,
        )
    if completed.stderr:
        return 2, _error_envelope(
            action=action,
            error_type="EngineProtocolError",
            message="validation engine emitted unexpected stderr",
            request_digest=request_digest,
        )
    try:
        engine_result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 2, _error_envelope(
            action=action,
            error_type="EngineProtocolError",
            message="validation engine did not emit one JSON document",
            request_digest=request_digest,
        )
    if not isinstance(engine_result, dict):
        return 2, _error_envelope(
            action=action,
            error_type="EngineProtocolError",
            message="validation engine JSON root is not an object",
            request_digest=request_digest,
        )
    status = engine_result.get("status")
    envelope: dict[str, object] = {
        "schema_version": "1",
        "status": "success"
        if completed.returncode == 0 and status == "success"
        else "rejected",
        "action": action,
        "request_digest": request_digest,
        "engine": {"name": "purged-kfold-validation", "version": __version__},
        "authoritative_cli_result": engine_result,
        "warnings": [
            "Structural leakage controls and model metrics are not profitability claims."
        ],
        "errors": [],
    }
    if envelope["status"] != "success":
        envelope["errors"] = [
            {
                "error_type": str(engine_result.get("error_type", "EngineRejected")),
                "message": str(
                    engine_result.get(
                        "message", "validation engine rejected the request"
                    )
                ),
            }
        ]
    return (0 if envelope["status"] == "success" else 2), envelope


def _materialize_example(output_directory: Path) -> tuple[str, ...]:
    if output_directory.exists():
        raise AgentRequestError("example output directory must not already exist")
    parent = output_directory.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=parent))
    names = ("features.csv", "manifest.json", "mapping.json")
    try:
        source = files("purged_kfold_validation").joinpath(
            "resources", "feature_upload", "examples", "raw"
        )
        for name in names:
            (staging / name).write_bytes(source.joinpath(name).read_bytes())
        request = _resource_json("request-v1.example.json")
        (staging / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (*names, "request.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "schema":
            _print_json(_resource_json(f"{args.kind}-v1.schema.json"))
            return 0
        if args.command == "example":
            output_directory = Path(args.output_dir)
            created = _materialize_example(output_directory)
            _print_json(
                {
                    "schema_version": "1",
                    "status": "success",
                    "action": "example",
                    "files": list(created),
                }
            )
            return 0
        if args.command == "demo":
            with tempfile.TemporaryDirectory(
                prefix="purged-cv-skill-demo-"
            ) as directory:
                target = Path(directory) / "input"
                _materialize_example(target)
                code, payload = _execute_request(target / "request.json")
                _print_json(payload)
                return code
        code, payload = _execute_request(Path(args.request).resolve())
        _print_json(payload)
        return code
    except (AgentRequestError, OSError, RuntimeError):
        _print_json(
            _error_envelope(
                action=str(args.command),
                error_type="AgentSkillError",
                message="Agent skill operation could not be completed",
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
