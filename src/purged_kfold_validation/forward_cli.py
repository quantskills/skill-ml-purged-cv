"""Installed JSON command for append-only temporal forward evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, NoReturn, Sequence, cast

from .errors import ForwardEvidenceError, ValidationError
from .temporal_forward import LocalTemporalForwardStore, TemporalForwardProtocol


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ForwardEvidenceError(message)


def _parser() -> _Parser:
    parser = _Parser(prog="purged-cv-forward")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "record", "settle", "status"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", required=True, type=Path)
        command.add_argument("--store", required=True, type=Path)
        if name in {"record", "settle"}:
            command.add_argument("--batch", required=True, type=Path)
        if name == "status":
            command.add_argument("--persist", action="store_true")
    return parser


def _read_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ForwardEvidenceError(f"{name} file does not exist")
    if resolved.stat().st_size > 16_777_216:
        raise ForwardEvidenceError(f"{name} file exceeds 16 MiB")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForwardEvidenceError(f"{name} JSON cannot be read: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ForwardEvidenceError(f"{name} JSON must be an object")
    return cast(Mapping[str, object], value)


def _protocol(path: Path) -> TemporalForwardProtocol:
    return TemporalForwardProtocol.from_mapping(_read_mapping(path, name="protocol"))


def _batch(path: Path, *, key: str) -> list[Mapping[str, object]]:
    value = _read_mapping(path, name="batch")
    if set(value) != {"schema_version", key} or value.get("schema_version") != "1":
        raise ForwardEvidenceError(
            f"batch fields must exactly equal schema_version and {key}"
        )
    rows = value[key]
    if not isinstance(rows, list) or not rows:
        raise ForwardEvidenceError(f"{key} must be a non-empty array")
    if len(rows) > 100_000 or any(not isinstance(item, Mapping) for item in rows):
        raise ForwardEvidenceError(f"{key} must contain at most 100000 objects")
    return [cast(Mapping[str, object], item) for item in rows]


def _fields(row: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(row) != expected:
        raise ForwardEvidenceError(
            f"{name} fields mismatch; expected {', '.join(sorted(expected))}"
        )


def _json(payload: object) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def main(argv: Sequence[str] | None = None) -> int:
    action: str | None = None
    try:
        args = _parser().parse_args(argv)
        action = str(args.command)
        protocol = _protocol(args.protocol)
        store = LocalTemporalForwardStore(args.store.resolve())
        if action == "init":
            store.register(protocol)
            result: object = store.report(protocol).canonical()
        elif action == "record":
            receipts = []
            expected = {
                "sample_id",
                "asset_id",
                "decision_session",
                "label_end_session",
                "label_available_at",
                "prediction",
                "feature_snapshot_digest",
            }
            for row in _batch(args.batch, key="predictions"):
                _fields(row, expected, name="prediction")
                receipts.append(
                    store.record_prediction(
                        protocol,
                        sample_id=row["sample_id"],  # type: ignore[arg-type]
                        asset_id=row["asset_id"],  # type: ignore[arg-type]
                        decision_session=row["decision_session"],  # type: ignore[arg-type]
                        label_end_session=row["label_end_session"],  # type: ignore[arg-type]
                        label_available_at=row["label_available_at"],  # type: ignore[arg-type]
                        prediction=row["prediction"],  # type: ignore[arg-type]
                        feature_snapshot_digest=row["feature_snapshot_digest"],  # type: ignore[arg-type]
                    ).digest
                )
            result = {
                "recorded_prediction_digests": receipts,
                "report": store.report(protocol).canonical(),
            }
        elif action == "settle":
            settlements = []
            expected = {"prediction_digest", "target", "target_source_digest"}
            for row in _batch(args.batch, key="settlements"):
                _fields(row, expected, name="settlement")
                settlements.append(
                    store.settle(
                        protocol,
                        prediction_digest=row["prediction_digest"],  # type: ignore[arg-type]
                        target=row["target"],  # type: ignore[arg-type]
                        target_source_digest=row["target_source_digest"],  # type: ignore[arg-type]
                    ).digest
                )
            result = {
                "recorded_settlement_digests": settlements,
                "report": store.report(protocol).canonical(),
            }
        else:
            report = store.report(protocol)
            if args.persist:
                store.persist_report(report)
            result = report.canonical()
        _json(
            {
                "schema_version": "1",
                "status": "success",
                "action": action,
                "result": result,
                "errors": [],
            }
        )
        return 0
    except (ValidationError, OSError, ValueError) as exc:
        _json(
            {
                "schema_version": "1",
                "status": "error",
                "action": action,
                "result": None,
                "errors": [{"code": "forward-evidence-rejected", "message": str(exc)}],
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
