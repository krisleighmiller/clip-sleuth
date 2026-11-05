"""Command-line interface for Clip Provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from clipprov import __version__
from clipprov.batch import BatchError, current_report_dir, match_directory
from clipprov.config import Config, resolve_speed_profile
from clipprov.index import IndexError as IndexReadError
from clipprov.index_batch import IndexBatchError, build_index_resumable
from clipprov.match import match_clip_path
from clipprov.media import MediaError

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2


def _atomic_write_json(path: str, text: str) -> None:
    from clipprov._io import atomic_text

    atomic_text(path, text)


@click.group()
@click.version_option(__version__, prog_name="clip-prov")
def main() -> None:
    """Clip Sleuth: match transformed clips against your local video library."""


@main.command("index")
@click.argument("library", type=click.Path(exists=True, file_okay=False))
@click.option("--out", "out", type=click.Path(), required=True, help="Output index directory.")
@click.option("--resume/--no-resume", default=True, help="Resume verified completed master shards.")
@click.option(
    "--continue-on-error/--fail-fast", default=False,
    help="Publish valid masters while reporting isolated failures.",
)
def index_cmd(library: str, out: str, resume: bool, continue_on_error: bool) -> None:
    """Build a resumable index from a library of source videos."""
    def progress(event):
        click.echo(json.dumps(event, sort_keys=True), err=True)

    try:
        manifest = build_index_resumable(
            Path(library), Path(out), Config(), resume=resume,
            continue_on_error=continue_on_error, progress=progress,
        )
        click.echo(json.dumps({
            "status": manifest["status"],
            "masters": len(manifest["masters"]),
            "failures": manifest.get("failures", []),
            "index": str(Path(out).resolve()),
        }, indent=2, sort_keys=True))
    except (IndexBatchError, IndexReadError, MediaError, OSError, ValueError, TypeError) as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)


@main.command("match")
@click.argument("clip", type=click.Path(exists=True))
@click.option("--index", "index_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out", "out", type=click.Path(), default=None, help="Write JSON result to file.")
@click.option(
    "--evidence-dir", type=click.Path(), default=None,
    help="Write auditable aligned frame pairs for the best candidate.",
)
@click.option(
    "--speed-profile", type=click.Choice(["wider-speed-v1"]), default=None,
    help="Opt in to the measured 0.5x-2.0x search profile.",
)
@click.option(
    "--geometry", "geometry_mode", type=click.Choice(["identity", "d4"]),
    default="identity", show_default=True,
    help="Opt in to mirror and quarter-turn descriptor hypotheses.",
)
@click.option(
    "--audio/--no-audio", "audio_enabled", default=False, show_default=True,
    help="Opt in to positive-only audio corroboration diagnostics.",
)
def match_cmd(
    clip: str, index_dir: str, out: str | None, evidence_dir: str | None,
    speed_profile: str | None, geometry_mode: str, audio_enabled: bool,
) -> None:
    """Match one immutable clip snapshot, using the same bytes for visual evidence."""
    from clipprov._io import digest
    from clipprov.eval import product_result
    from clipprov.index import _staged_source_snapshot, config_from_manifest, load_index

    try:
        original_clip = Path(clip).resolve()
        with _staged_source_snapshot(original_clip) as (staged_clip, clip_digest):
            matching_receipt = None
            if speed_profile is None and geometry_mode == "identity" and not audio_enabled:
                diag, manifest = match_clip_path(
                    staged_clip, Path(index_dir), geometry_mode=geometry_mode
                )
            else:
                loaded_index = load_index(Path(index_dir))
                match_config, profile_receipt = resolve_speed_profile(
                    speed_profile, config_from_manifest(loaded_index[0])
                )
                match_config = match_config.model_copy(update={
                    "match": match_config.match.model_copy(update={
                        "geometry_mode": geometry_mode,
                        "audio_enabled": audio_enabled,
                    })
                })
                diag, manifest = match_clip_path(
                    staged_clip, Path(index_dir), match_config,
                    loaded_index=loaded_index,
                )
                matching_receipt = {
                    "schema_version": 2,
                    "speed_profile": profile_receipt,
                    "geometry_mode": geometry_mode,
                    "audio_enabled": audio_enabled,
                    "input_receipt": {
                        "clip_sha256": clip_digest,
                        "index_manifest_sha256": manifest.manifest_sha256,  # type: ignore[attr-defined]
                    },
                }
            result = product_result(
                clip, diag, manifest, matching_receipt=matching_receipt
            )
            if evidence_dir:
                from clipprov.evidence import generate_evidence, validate_evidence

                source = result["source_record"]
                if source is None:
                    raise ValueError("no candidate source is available for evidence")
                master_path = Path(source["original_root_hint"]) / source["source_relpath"]
                evidence = generate_evidence(
                    staged_clip, master_path, master_id=source["id"],
                    source_digest=source["content_digest"],
                    aligned_timestamps=result["confidence"]["timing"]["aligned_timestamps"],
                    output_dir=Path(evidence_dir),
                )
                evidence["clip_path"] = str(original_clip)
                result = product_result(
                    str(original_clip), diag, manifest, evidence=evidence,
                    matching_receipt=matching_receipt,
                )
                problems = validate_evidence(evidence, result)
                if problems:
                    raise ValueError("invalid evidence: " + "; ".join(problems))
            if digest(Path(clip)) != clip_digest:
                raise ValueError("clip changed during matching")
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if out:
            _atomic_write_json(out, text)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)
    click.echo(text, nl=False)
    if not diag.accepted:
        sys.exit(EXIT_NO_MATCH)


@main.command("match-dir")
@click.argument("clips", type=click.Path(exists=True, file_okay=False))
@click.option("--index", "index_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out", "out_dir", type=click.Path(), required=True)
@click.option("--resume/--no-resume", default=True)
@click.option("--continue-on-error/--fail-fast", default=True)
@click.option("--evidence/--no-evidence", default=True)
@click.option("--calibration", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--speed-profile", type=click.Choice(["wider-speed-v1"]), default=None,
    help="Opt in to the measured 0.5x-2.0x search profile.",
)
@click.option(
    "--geometry", "geometry_mode", type=click.Choice(["identity", "d4"]),
    default="identity", show_default=True,
    help="Opt in to mirror and quarter-turn descriptor hypotheses.",
)
@click.option(
    "--audio/--no-audio", "audio_enabled", default=False, show_default=True,
    help="Opt in to positive-only audio corroboration diagnostics.",
)
def match_dir_cmd(
    clips, index_dir, out_dir, resume, continue_on_error, evidence, calibration,
    speed_profile, geometry_mode, audio_enabled,
):
    """Match a directory deterministically with progress, resume, and offline reports."""
    def progress(event):
        click.echo(json.dumps(event, sort_keys=True), err=True)

    try:
        report = match_directory(
            clips, index_dir, out_dir, resume=resume,
            continue_on_error=continue_on_error, evidence=evidence,
            calibration_path=calibration, progress=progress,
            speed_profile=speed_profile,
            geometry_mode=geometry_mode,
            audio_enabled=audio_enabled,
        )
        report_dir = current_report_dir(Path(out_dir))
    except (BatchError, IndexReadError, MediaError, OSError, ValueError, TypeError) as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)
    click.echo(json.dumps({
        "report": str((report_dir / "report.json").resolve()),
        "results": len(report["results"]),
        "failures": len(report["failures"]),
    }, indent=2, sort_keys=True))
    if report["failures"]:
        sys.exit(EXIT_NO_MATCH)


@main.command("eval")
@click.option("--results", required=True, type=click.Path(exists=True))
@click.option("--truth", required=True, type=click.Path(exists=True))
@click.option("--out", default=None, type=click.Path())
@click.option("--tolerance", default=1.5, type=float)
def eval_cmd(results, truth, out, tolerance):
    """Score product match results against ground truth."""
    from clipprov.eval import load_product_results, load_truth, score_product

    try:
        results_rows, results_sha = load_product_results(Path(results), with_digest=True)
        truth_map, truth_sha = load_truth(Path(truth), with_digest=True)

        scored = score_product(
            results_rows,
            truth_map,
            tolerance_seconds=tolerance,
            results_digest=results_sha,
            truth_digest=truth_sha,
        )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)
    text = json.dumps(scored, indent=2, sort_keys=True) + "\n"
    if out:
        try:
            _atomic_write_json(out, text)
        except OSError as exc:
            click.echo(f"error: {exc}", err=True)
            sys.exit(EXIT_ERROR)
    click.echo(text, nl=False)


@main.command("generate")
@click.argument("library", type=click.Path(exists=True, file_okay=False))
@click.option("--out", required=True, type=click.Path())
def generate_cmd(library, out):
    """Generate frozen development, holdout and source-withheld controls."""
    from clipprov.harness import generate_controls

    try:
        manifest = generate_controls(Path(library), Path(out))
        click.echo(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)


@main.command("demo-corpus")
@click.argument("out", type=click.Path())
def demo_corpus_cmd(out):
    """Create a portable three-source evaluation smoke corpus."""
    from clipprov.harness import generate_demo_corpus

    try:
        receipt = generate_demo_corpus(Path(out))
        click.echo(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)


@main.command("evaluate")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.option("--index", "index_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out", required=True, type=click.Path())
@click.option("--library", default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--benchmark-results", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--benchmark-truth", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--shortlist-samples", default=8, type=click.IntRange(0, 64))
@click.option("--tolerance", default=1.5, type=float)
def evaluate_cmd(
    manifest,
    index_dir,
    out,
    library,
    benchmark_results,
    benchmark_truth,
    shortlist_samples,
    tolerance,
):
    """Audit frozen controls and emit benchmark plus per-group metrics."""
    from clipprov.harness import evaluate_controls

    try:
        report = evaluate_controls(
            manifest,
            index_dir,
            out,
            library=library,
            benchmark_results=benchmark_results,
            benchmark_truth=benchmark_truth,
            shortlist_samples=shortlist_samples,
            tolerance=tolerance,
        )
        click.echo(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
