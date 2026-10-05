# Project Status

## Current phase

**PR [#16](https://github.com/QuitQui/in-n-out/pull/16)** — push an oversized
HuggingFace dataset to Drive. Open, 5 commits, 145 tests passing, awaiting
review. Not merged.

Open question: a 100 GB transfer over an unreliable link is still unproven
end to end. The tooling is built and verified against the live Drive API in
the small, but no full-size run has completed yet.

## Recently completed

- `--repo-type model|dataset|space` for `--hf` — datasets live in a separate
  Hub namespace, so a dataset ID resolved as a model 404s.
- Nested Drive folder paths (`--drive "A/B"`), each level matched *within its
  parent*. Fixed two bugs: `files().create` omitted `parents` so every folder
  landed at My Drive root, and the folder lookup matched names Drive-wide and
  took `files[0]`.
- `innout-batch` (`innout/batch.py`) — walks a repo file by file so peak disk
  is ~2x the largest single file instead of ~4x the whole dataset. Resumable
  via a ledger at `~/.innout_batches/<repo>.json`; `pull` restores original
  filenames and verifies sha256.
- Outer retry with exponential backoff on every Drive call.
  `googleapiclient`'s `num_retries` only guards the request that *opens* a
  resumable upload, not the PUTs carrying the bytes, so any dropped
  connection aborted a multi-hour transfer.
- Paginated Drive listings. Drive caps a listing at 100 files silently; a
  truncated chunk list joins into an undecryptable blob.
- Flat layout: a file's chunks go in the folder mirroring its repo path, so
  many pushes share a folder and are told apart by the session ID in each
  chunk's filename. `pull` filters on it; an ambiguous folder refuses.
- Per-file failure isolation in `innout-batch`: record under `failures`,
  delete the orphan chunks, continue, exit non-zero with the retry list.

## Scheduled / next

1. Complete a full-size dataset transfer and attach the result to PR #16.
2. Then merge #16 via GitHub (never locally).

## Known issues

- **Console scripts cannot import `innout` in this venv.** `.pth` files are
  not processed by the interpreter at all — virtualenv's own `_virtualenv.pth`
  does not load either, while `site` is imported and `getsitepackages()`
  returns the right directory. `uv sync` does not fix it. Affects `innout`,
  `innout-server` and `innout-batch` equally. Workaround everywhere, including
  the README: `uv run python -m innout.<module>`. Unrelated to PR #16;
  may need the venv rebuilt.
