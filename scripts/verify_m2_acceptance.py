#!/usr/bin/env python3
"""Derive the M2 verdict from immutable evidence; never trust a claimed verdict."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_SLOT_TRANSITIONS = {
    ("EMPTY", "PREPARING"),
    ("PREPARING", "READY"),
    ("READY", "COPYING"),
    ("COPYING", "RESIDENT"),
    ("RESIDENT", "IN_USE"),
    ("IN_USE", "RETIRED"),
    ("RETIRED", "EMPTY"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path}: root is not an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_file(root: Path) -> None:
    sums = root / "SHA256SUMS"
    require(sums.is_file(), "SHA256SUMS is missing")
    seen: set[str] = set()
    for line_number, line in enumerate(
        sums.read_text(encoding="utf-8").splitlines(), 1
    ):
        require("  " in line, f"SHA256SUMS:{line_number}: malformed line")
        expected, relative = line.split("  ", 1)
        require(len(expected) == 64, f"SHA256SUMS:{line_number}: invalid digest")
        require(relative not in seen, f"SHA256SUMS: duplicate path {relative}")
        seen.add(relative)
        path = root / relative
        require(path.is_file(), f"missing hashed artifact: {relative}")
        require(sha256(path) == expected, f"SHA-256 mismatch: {relative}")


def range_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("phase"),
        record.get("graph_index"),
        record.get("split_id"),
        record.get("slot_id"),
        record.get("slot_generation"),
        record.get("offset_in_tensor"),
        record.get("bytes"),
    )


def verify_trace(path: Path, controller: dict[str, Any], run: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    bound: Counter[tuple[Any, ...]] = Counter()
    committed: Counter[tuple[Any, ...]] = Counter()
    transitions: list[tuple[int, tuple[Any, ...], str, str, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            require(isinstance(record, dict), f"{path}:{line_number}: not an object")
            event = record.get("event")
            require(isinstance(event, str), f"{path}:{line_number}: missing event")
            counts[event] += 1

            if event == "PREPARED_RANGE_BOUND":
                bound[range_key(record)] += 1
            elif event == "PREPARED_RANGE_COMMITTED":
                require(
                    record.get("prepared_source_consumed_before_retire") is True,
                    f"run {run}:{line_number}: range retired before consumption",
                )
                require(
                    record.get("copy_submission") == "SYNCHRONOUS_TENSOR_SET",
                    f"run {run}:{line_number}: unapproved prepared copy submission",
                )
                committed[range_key(record)] += 1
            elif event == "SLOT_STATE":
                before = record.get("from_state")
                after = record.get("to_state")
                require(
                    (before, after) in ALLOWED_SLOT_TRANSITIONS,
                    f"run {run}:{line_number}: illegal slot transition {before}->{after}",
                )
                slot_key = (
                    record.get("supply_phase"),
                    record.get("slot_id"),
                    record.get("slot_generation"),
                )
                sequence = record.get("transition_sequence")
                require(
                    isinstance(sequence, int) and sequence > 0,
                    f"run {run}:{line_number}: missing authoritative transition sequence",
                )
                transitions.append((sequence, slot_key, before, after, line_number))

    graph_count = controller["graph_count"]
    prepared_count = controller["prepared_range_count"]
    require(
        counts["GRAPH_IDENTITY_VERIFIED"]
        + counts["GRAPH_REUSE_IDENTITY_VERIFIED"]
        == graph_count,
        f"run {run}: graph identity event count mismatch",
    )
    require(
        counts["GRAPH_COMPLETE_AND_LEASES_RETIRED"] == graph_count,
        f"run {run}: graph completion count mismatch",
    )
    for event in (
        "PHASE_SUPPLY_START",
        "PHASE_SUPPLY_COMPLETE",
    ):
        require(counts[event] == graph_count, f"run {run}: {event} count mismatch")
    for event in (
        "SCHEDULER_DECISION",
        "PREPARED_RANGE_BOUND",
        "PREPARED_RANGE_COMMITTED",
    ):
        require(counts[event] == prepared_count, f"run {run}: {event} count mismatch")
    require(bound == committed, f"run {run}: prepared range bind/commit mismatch")
    expected_transition_count = prepared_count * len(ALLOWED_SLOT_TRANSITIONS)
    require(
        len(transitions) == expected_transition_count,
        f"run {run}: Ring transition count mismatch",
    )
    transitions.sort(key=lambda item: item[0])
    require(
        [item[0] for item in transitions] == list(range(1, expected_transition_count + 1)),
        f"run {run}: transition sequence is not contiguous",
    )
    last_slot_state: dict[tuple[Any, ...], str] = {}
    completed_slot_lifecycles = 0
    for _, slot_key, before, after, line_number in transitions:
        previous = last_slot_state.get(slot_key, "EMPTY")
        require(
            previous == before,
            f"run {run}:{line_number}: authoritative slot sequence is discontinuous "
            f"{previous}!={before}",
        )
        last_slot_state[slot_key] = after
        if after == "EMPTY":
            completed_slot_lifecycles += 1
    require(
        completed_slot_lifecycles == prepared_count,
        f"run {run}: incomplete Ring slot lifecycle",
    )
    require(
        all(state == "EMPTY" for state in last_slot_state.values()),
        f"run {run}: Ring slot not EMPTY at trace end",
    )
    require(counts["RUNTIME_SOURCE_FALLBACK"] == 0, f"run {run}: fallback event")
    return counts


def verify_controller(controller: dict[str, Any], run: int) -> None:
    graph_count = controller["graph_count"]
    ranges = controller["prepared_range_count"]
    supplied_bytes = controller["prepared_range_bytes"]

    require(graph_count == 128, f"run {run}: graph count is not 128")
    require(
        controller["graph_signature_match_count"] == graph_count
        and controller["graph_demand_match_count"] == graph_count,
        f"run {run}: graph contract mismatch",
    )
    require(ranges > 0 and supplied_bytes > 0, f"run {run}: no prepared supply")
    require(
        controller["runtime_source_fallback_count"] == 0,
        f"run {run}: ordinary Runtime source fallback occurred",
    )
    require(
        controller["model_store_mode"] == "read",
        f"run {run}: Model Store is not explicit range-read mode",
    )
    require(
        controller["model_store_bytes_read"]
        == controller["pipeline_bytes_prepared"]
        == controller["pipeline_bytes_supplied"]
        == supplied_bytes
        == controller["weight_copy_bytes"],
        f"run {run}: Model Store / pipeline / backend byte accounting mismatch",
    )
    require(
        controller["pipeline_phases_started"]
        == controller["pipeline_phases_completed"]
        == graph_count,
        f"run {run}: pipeline phase accounting mismatch",
    )
    require(
        controller["scheduler_decisions"]
        == controller["chunks_prepared"]
        == controller["chunks_retired"]
        == ranges,
        f"run {run}: Scheduler / Ring range accounting mismatch",
    )
    require(
        controller["prefetch_hits"] + controller["consumer_waits"] == ranges,
        f"run {run}: Queue acquisition accounting mismatch",
    )
    require(
        1 <= controller["ring_peak_in_flight"] <= controller["ring_depth"] < ranges,
        f"run {run}: Ring is not finite or exceeded its depth",
    )
    require(
        controller["peak_ready_queue"] <= controller["ring_depth"],
        f"run {run}: Ready Queue exceeded Ring depth",
    )
    require(
        controller["ring_illegal_transition_attempts"] == 0,
        f"run {run}: illegal Ring transition recorded",
    )
    require(
        controller["pipeline_failed"] is False and controller["failed"] is False,
        f"run {run}: supply pipeline failed",
    )


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(
        "results/m2-gemma4-12b-real-supply-repair-20260804"
    )
    require(root.is_dir(), f"evidence root not found: {root}")

    retraction_path = root / "M2_RETRACTION.json"
    if retraction_path.is_file():
        retraction = load_json(retraction_path)
        raise RuntimeError(
            "evidence verdict was withdrawn: "
            f"{retraction.get('reason', 'see M2_RETRACTION.json')}"
        )

    verify_sha_file(root)
    summary = load_json(root / "M2_ACCEPTANCE_SUMMARY.json")
    index = load_json(root / "M2_EVIDENCE_INDEX.json")
    implementation = load_json(root / "IMPLEMENTATION_IDENTITY.json")
    baseline = load_json(root / "preflight/M0_TOKEN_ID_BASELINE.json")
    manifest_path = root / "manifest/M2_ACCEPTANCE_MANIFEST.json"
    repeat_manifest_path = root / "manifest/M2_ACCEPTANCE_MANIFEST_REPEAT.json"
    manifest = load_json(manifest_path)

    require(summary["mission"] == "ALSR-M2-M5-REAL-SUPPLY-REPAIR-01", "mission mismatch")
    require(index["mission"] == summary["mission"], "index mission mismatch")
    require(sha256(manifest_path) == sha256(repeat_manifest_path), "manifest pair differs")
    require(manifest["integrity"]["complete"] is True, "manifest incomplete")

    runtime_patch = summary["identity"]["runtime_patch_sha256"]
    require(
        manifest["identity"]["runtime_patch_sha256"] == runtime_patch,
        "manifest/runtime patch identity mismatch",
    )
    require(
        implementation["runtime_patch_sha256"] == runtime_patch,
        "implementation/runtime patch identity mismatch",
    )
    for artifact in implementation["artifacts"]:
        artifact_path = root / artifact["path"]
        require(artifact_path.is_file(), f"missing implementation artifact: {artifact['path']}")
        require(
            sha256(artifact_path) == artifact["sha256"],
            f"implementation artifact hash mismatch: {artifact['path']}",
        )
    capture_hashes = {
        sha256(root / "identity_capture_1/GRAPH_TRACE.json"),
        sha256(root / "identity_capture_2/GRAPH_TRACE.json"),
    }
    require(
        capture_hashes == {summary["identity"]["capture_graph_trace_sha256"]},
        "identity captures are not byte-identical",
    )

    elapsed: list[float] = []
    for run in range(1, 6):
        run_root = root / f"acceptance_128_run_{run}"
        result = load_json(run_root / "RESULT.json")
        graph_trace = load_json(run_root / "GRAPH_TRACE.json")
        correctness = result["correctness"]
        controller = result["supply_controller"]
        inventory = result["model_buffer_inventory"]
        resources = result["process_resources"]
        runtime = graph_trace["runtime"]

        require(result["mode"] == "alsr", f"run {run}: not ALSR mode")
        require(result["generated_token_count"] == 128, f"run {run}: token count")
        require(
            result["generated_token_ids"] == baseline["generated_token_ids"],
            f"run {run}: token ID mismatch",
        )
        require(result["generated_text"] == baseline["response_text"], f"run {run}: text mismatch")
        require(correctness["accepted"] is True, f"run {run}: correctness rejected")
        require(
            runtime["callback_can_request_prepared_ranges"] is True
            and runtime["callback_requests_tensor_data"] is True,
            f"run {run}: callback did not request prepared data",
        )
        require(result["runtime_patch_sha256"] == runtime_patch, f"run {run}: patch mismatch")
        require(runtime["patch_sha256"] == runtime_patch, f"run {run}: trace patch mismatch")
        require(
            sha256(run_root / "GRAPH_TRACE.json")
            == summary["identity"]["accepted_graph_trace_sha256"],
            f"run {run}: accepted graph trace identity mismatch",
        )
        require(
            inventory["hidden_second_full_load_detected"] is False
            and inventory["cpu_repack_seen"] is False,
            f"run {run}: hidden load/repack detected",
        )
        require(resources["process_elapsed_seconds"] < 1800, f"run {run}: timeout")
        verify_controller(controller, run)
        verify_trace(run_root / "supply_trace.jsonl", controller, run)
        recorded = summary["runs"][run - 1]
        require(
            recorded["run"] == run
            and recorded["seconds"] == resources["process_elapsed_seconds"]
            and recorded["peak_rss_bytes"] == resources["peak_resident_set_bytes"]
            and recorded["model_store_bytes_read"] == controller["model_store_bytes_read"]
            and recorded["prepared_ranges"] == controller["prepared_range_count"],
            f"run {run}: summary row mismatch",
        )
        elapsed.append(resources["process_elapsed_seconds"])

    ordered = sorted(elapsed)
    aggregate = summary["aggregate"]
    require(aggregate["accepted_run_count"] == 5, "summary run count mismatch")
    require(
        aggregate["seconds_min"] == ordered[0]
        and aggregate["seconds_median"] == ordered[2]
        and aggregate["seconds_max"] == ordered[-1],
        "summary timing aggregation mismatch",
    )
    print(
        "M2_ACCEPTANCE_VERIFIED: GREEN "
        "(derived: 5/5 exact, explicit Model Store -> finite Ring -> Backend, fallback 0)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"M2_ACCEPTANCE_VERIFICATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
