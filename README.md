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
| Acquire | `sources.py` | URL download, local path, GitHub clone, HuggingFace snapshot |
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
  --server <url> \
  [--passphrase <str>] \
  [--api-key <key>] \
  [--chunk-size <MB>]
```

Source options (pick one):

| Flag | Description |
|------|-------------|
| `--url <url>` | Download from a remote URL |
| `--local <path>` | Local file or directory (directories are tar.gz'd) |
| `--github owner/repo[@branch]` | Clone a GitHub repo |
| `--hf org/model` | HuggingFace snapshot |

If `--passphrase` is omitted, checks `INNOUT_PASSPHRASE` env var, then prompts interactively.
If `--api-key` is omitted, checks `INNOUT_API_KEY` env var.

`--chunk-size` defaults to 1800 MB. Tune it to stay under your server's upload limits.

On success, prints the session ID — save it, you need it to pull.

---

## Pull

```bash
innout pull <session_id> \
  --server <url> \
  [--passphrase <str>] \
  [--api-key <key>] \
  [--output <dir>]
```

Downloads all parts for the session, joins them, decrypts, and writes the result to `<output>/result`. If the original source was a directory, `result` is a `.tar.gz` archive — unpack it with `tar xzf`.

---

## End-to-end example: Tiny ImageNet

[Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) is a 237 MB public dataset — 200 classes, 100k training images. Useful for testing the full pipeline without ImageNet registration.

**Start the server:**

```bash
export INNOUT_API_KEY="demo-key-change-me"
uv run innout-server --store /tmp/innout-store --port 8765
```

**Push:**

```bash
uv run python -m innout.cli push \
  --url http://cs231n.stanford.edu/tiny-imagenet-200.zip \
  --server http://localhost:8765 \
  --passphrase "my-passphrase" \
  --api-key demo-key-change-me \
  --chunk-size 50

# Done. Session ID: 41825928-771f-4e21-ad46-4fa6eb79b202  Parts: 5
```

**Check the manifest:**

```bash
curl -s -H "Authorization: Bearer demo-key-change-me" \
  http://localhost:8765/manifest/41825928-771f-4e21-ad46-4fa6eb79b202 | python3 -m json.tool

{
    "session_id": "41825928-771f-4e21-ad46-4fa6eb79b202",
    "parts": ["000", "001", "002", "003", "004"]
}
```

**Pull and recover:**

```bash
uv run python -m innout.cli pull \
  41825928-771f-4e21-ad46-4fa6eb79b202 \
  --server http://localhost:8765 \
  --passphrase "my-passphrase" \
  --api-key demo-key-change-me \
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
