# Agent First-Class Plan

## Goal

Make `txt2audio` a first-class tool for LLM agents, not only a human CLI. The target is a tool surface that is discoverable, schema-bound, deterministic in non-interactive mode, explicit about side effects, and easy for agents to recover from when a run fails.

## Design Principles

- Keep stdout machine-readable in machine modes; keep human progress and warnings on stderr.
- Provide JSON Schema contracts for every public JSON result.
- Use stable error codes with retryability and structured details.
- Separate planning/validation from side-effectful generation.
- Make long-running work observable without corrupting the final JSON result.
- Treat output files as artifacts with metadata, not just paths.
- Preserve idempotent resume behavior and document destructive cleanup.
- Keep the shell CLI as the base interface, then expose the same contract through an MCP wrapper.

## Current Surface

- `--json` returns a final run result.
- `--plan-json` returns a model-free generation plan.
- `--validate-paths --json` validates parsed output paths.
- `--quiet` suppresses successful human output.
- `--chapter-manifest` writes a per-chapter JSON manifest.
- Atomic writes and existing-output skips already make reruns mostly idempotent.

## Target Contract

Public JSON outputs:

- Error envelope: `schemas/error.schema.json`
- Run result: `schemas/run-result.schema.json`
- Plan result: `schemas/plan-result.schema.json`
- Path validation result: `schemas/validate-paths-result.schema.json`
- Chapter manifest: `schemas/chapter-manifest.schema.json`

Future event stream:

- `--events-jsonl PATH|-` emits JSON Lines events without mixing them into the final stdout JSON.
- Events include `run_started`, `chapter_started`, `clip_generated`, `artifact_created`, `chapter_completed`, `warning`, `error`, and `run_completed`.

Future MCP tools:

- `txt2audio_validate_book`: read-only, model-free.
- `txt2audio_plan_conversion`: read-only, model-free.
- `txt2audio_convert_book`: side-effectful, idempotent when resume is enabled.
- `txt2audio_get_manifest`: read-only artifact lookup.

## Roadmap

### P0: Stable Machine Contract

- Add schema files for current public JSON outputs.
- Add `schema_version` to run, plan, validation, and error results.
- Keep backward-compatible `error` while adding `error_code`.
- Add `retryable` and `details` to error results.
- Add lightweight contract tests for schema files and core output builders.

### P1: Artifact Metadata

- Replace plain generated path lists with artifact objects while keeping legacy path lists during transition.
- Include `path`, `format`, `bytes`, `chapter_index`, `clip_index`, `role`, and optional `duration_seconds`.
- Write `chapter_manifest.json` by default in JSON/agent mode, or return enough metadata to avoid directory scans.

### P2: Event Stream

- Add `--events-jsonl PATH|-`.
- Keep final `--json` result as a single stdout JSON object.
- Emit event records with `schema_version`, `event`, `run_id`, `time`, and structured payloads.

### P3: MCP Wrapper

- Add a small MCP server wrapper around the CLI/core functions.
- Publish tool input/output schemas from the same `schemas/` contracts.
- Annotate tools as read-only, destructive, idempotent, or open-world where appropriate.

## Acceptance Criteria

- Every documented JSON mode has a schema and tests.
- Invalid input, missing dependencies, TTS failures, and media conversion failures use stable error envelopes.
- Agents can plan, validate, run, and inspect outputs without parsing human text.
- Long-running runs can be monitored without mixing progress into final JSON.
- Human CLI behavior remains readable and quiet mode remains quiet on success.
