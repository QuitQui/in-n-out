# in-n-out

Encrypt a file or repo locally, split it into fixed-size chunks, upload to your own server, and pull it back anywhere. Nothing leaves your machine in plaintext.

```
innout push --url https://example.com/dataset.zip \
            --server http://your-server:8000 \
            --passphrase "correct horse battery"
# Done. Session ID: 41825928-771f-4e21-ad46-4fa6eb79b202  Parts: 5

innout pull 41825928-771f-4e21-ad46-4fa6eb79b202 \
            --server http://your-server:8000 \
            --passphrase "correct horse battery" \
            --output ./recovered
# Done. Output: ./recovered/result
```

---

## What it does

| Step | Tool | Detail |
|------|------|--------|
| Acquire | `sources.py` | URL download, local path, GitHub clone, Hugging Face snapshot |
| Encrypt | `crypto.py` | AES-256-GCM, PBKDF2-HMAC-SHA256 (100k iterations), random salt+nonce per file |
| Split | `splitter.py` | Fixed-size binary chunks, zero-padded part numbers (`000`, `001`, …) |
| Upload | `uploader.py` | Multipart POST with retry and tqdm progress |
| Serve | `server.py` | Flask REST API — upload, manifest, download |
| Pull | `uploader.py` | Fetches manifest → downloads chunks → joins → decrypts |

The server stores nothing except opaque binary blobs indexed by UUID session ID and part number. It has no concept of what the data is.

---

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/QuitQui/in-n-out.git
cd in-n-out
uv sync
```

Verify:

```bash
uv run python -m innout.cli --help
uv run innout-server --help
```

---

## Running the server

The server requires an API key. Every endpoint checks `Authorization: Bearer <key>`.

```bash
export INNOUT_API_KEY="your-secret-key"
uv run innout-server --store /data/chunks --port 8000
```

Options:

```
--store <dir>   Where to persist chunk files (default: /tmp/innout-store)
--port  <n>     Port to listen on (default: 8000)
--host  <addr>  Bind address (default: 0.0.0.0)
--api-key <key> Override INNOUT_API_KEY env var
--rate-limit-storage-uri <uri> Override INNOUT_LIMITER_STORAGE_URI env var (default: memory://)
```

The server enforces:
- Bearer token auth on every endpoint
- UUID validation on session IDs (prevents path traversal)
- 3-digit validation on part numbers
- 2 GB per-chunk upload cap
- Per-IP rate limiting (200 req/min global, 60 req/min on `/upload`)

To host publicly, put it behind nginx or Caddy with TLS. The Flask dev server is fine for local use.

---

## Push

```bash
innout push [source] \
  --server <url>            # custom server, OR
  --drive <folder>          # Google Drive folder name
  [--passphrase <str>] \
  [--credentials <path>] \
  [--api-key <key>] \
  [--chunk-size <MB>]
```

Source options (pick one):

| Flag | Description |
|------|-------------|
| `--url <url>` | Download from a remote URL |
| `--local <path>` | Local file or directory (directories are tar.gz'd) |
| `--github owner/repo[@branch]` | Clone a GitHub repo |
| `--hf org/model` | Hugging Face snapshot |

If `--passphrase` is omitted, checks `INNOUT_PASSPHRASE` env var, then prompts interactively.
If `--api-key` is omitted, checks `INNOUT_API_KEY` env var.

`--chunk-size` defaults to 1800 MB. Tune it to stay under your server's upload limits.

### Push to Google Drive

```bash
innout push --local ./my-project \
  --drive "my-backup-folder" \
  --credentials credentials.json \
  --passphrase "correct horse battery"
# Done. Session ID: ...  Parts: 3
# Drive folder: https://drive.google.com/drive/folders/...
```

`--credentials` defaults to `credentials.json` in the current directory. Download it from Google Cloud Console → APIs & Services → OAuth 2.0 Client IDs. On first run, a browser window opens for OAuth consent; the token is cached at `~/.innout_drive_token.json` for future runs.

On success, prints the session ID and the Drive folder URL — save both.

---

## Pull

### Pull from custom server

```bash
innout pull <session_id> \
  --server <url> \
  [--passphrase <str>] \
  [--api-key <key>] \
  [--output <dir>]
```

Downloads all parts for the session, joins them, decrypts, and writes the result to `<output>/result`.

### Pull from Google Drive

```bash
innout pull \
  --drive "my-backup-folder" \
  --credentials credentials.json \
  --passphrase "correct horse battery" \
  --output ./recovered
# Done. Output: ./recovered/result
```

`--drive` and `--server` are mutually exclusive. The Drive folder name must match the one used during `push`.

### Pull from a manually downloaded directory

If you downloaded the chunk files by hand (e.g. via the Drive web UI), point `--from-dir` at the local folder:

```bash
innout pull \
  --from-dir ./downloaded-chunks \
  --passphrase "correct horse battery" \
  --output ./recovered
# Done. Output: ./recovered/result
```

Chunk files must match the pattern `*.part???` (e.g. `session.part000`, `session.part001`).

If the original source was a directory, `result` is a `.tar.gz` archive — unpack it with `tar xzf`.

---

## End-to-end example: Tiny ImageNet

[Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) is a 237 MB public dataset — 200 classes, 100k training images. Useful for testing the full pipeline without ImageNet registration.

**Start the server:**

```bash
export INNOUT_API_KEY="$(openssl rand -hex 32)"
uv run innout-server --store /tmp/innout-store --port 8765
```
(If `openssl` is unavailable, generate a key with Python: `python3 -c "import secrets; print(secrets.token_hex(32))"`.)
Verify the key length is 64 characters: `python3 -c "import os; print(len(os.environ.get('INNOUT_API_KEY','')))"`.

**Push:**

```bash
uv run python -m innout.cli push \
  --url http://cs231n.stanford.edu/tiny-imagenet-200.zip \
  --server http://localhost:8765 \
  --passphrase "my-passphrase" \
  --api-key "$INNOUT_API_KEY" \
  --chunk-size 50

# Done. Session ID: 41825928-771f-4e21-ad46-4fa6eb79b202  Parts: 5
```

**Check the manifest:**

```bash
# Run an inline Python script to fetch and print the manifest JSON.
python3 - <<'PY'
import json
import os
import uuid
import requests

api_key = os.environ.get("INNOUT_API_KEY")
if not api_key or not api_key.strip():
    raise SystemExit("INNOUT_API_KEY is not set")

session_id = os.environ.get("INNOUT_SESSION_ID")
if not session_id or not session_id.strip():
    raise SystemExit("Set INNOUT_SESSION_ID to the Session ID printed by `push`")
try:
    session_id = str(uuid.UUID(session_id))
except ValueError as exc:
    raise SystemExit("INNOUT_SESSION_ID must be a UUID") from exc
url = f"http://localhost:8765/manifest/{session_id}"
resp = requests.get(
    url,
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=10,
)
resp.raise_for_status()
print(json.dumps(resp.json(), indent=2))
PY
```
(`requests` is already installed with `uv sync` from this project.)

**Pull and recover:**

```bash
uv run python -m innout.cli pull \
  41825928-771f-4e21-ad46-4fa6eb79b202 \
  --server http://localhost:8765 \
  --passphrase "my-passphrase" \
  --api-key "$INNOUT_API_KEY" \
  --output ./recovered

# Done. Output: ./recovered/result

# result is the original zip — verify it:
python3 -c "import zipfile; print(zipfile.is_zipfile('recovered/result'))"
# True
```

---

## Server API reference

All endpoints require `Authorization: Bearer <INNOUT_API_KEY>`.

| Method | Path | Body / Params | Returns |
|--------|------|---------------|---------|
| `POST` | `/upload` | Form: `session_id`, `part`, `total_parts`; File: `file` | `{"status":"ok","session_id":"...","part":"..."}` |
| `GET` | `/manifest/<session_id>` | — | `{"session_id":"...","parts":["000","001",...]}` |
| `GET` | `/download/<session_id>/<part>` | — | Binary chunk data |
| `GET` | `/` | — | Server status |

---

## Security model

- **Encryption**: AES-256-GCM with a per-file random 16-byte salt and 12-byte nonce. The GCM tag authenticates the entire ciphertext — any corruption or tampering raises `ValueError` on decrypt.
- **Key derivation**: PBKDF2-HMAC-SHA256, 100,000 iterations. The passphrase never leaves the client.
- **Server auth**: Constant-time Bearer token comparison (`hmac.compare_digest`) prevents timing attacks.
- **Path traversal**: Session IDs validated as UUIDs; part numbers validated as exactly three digits.
- **Rate limiting**: Flask-Limiter enforces per-IP caps. Back this with nginx rate limiting in production.
- **Data opacity**: The server stores raw binary blobs. It has no passphrase and cannot decrypt anything.

---

## Development

```bash
uv sync
uv run pytest
```

Tests live in `tests/`. Each module has a corresponding test file:

```
tests/test_crypto.py    # roundtrip, wrong passphrase, tampered ciphertext
tests/test_splitter.py  # split/join, empty file, chunk ordering
tests/test_server.py    # upload, manifest, download, auth, validation
```

To add a dependency: `uv add <pkg>`. Never use `pip install` directly.
