import argparse
import getpass
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from innout import crypto, sources, splitter, uploader


def get_passphrase(args_passphrase: str | None) -> str:
    if args_passphrase:
        return args_passphrase
    env_pass = os.environ.get("INNOUT_PASSPHRASE")
    if env_pass:
        return env_pass
    return getpass.getpass("Passphrase: ")


def cmd_push(args: argparse.Namespace) -> None:
    if not args.server and not args.drive:
        raise SystemExit("error: one of --server or --drive is required")

    if args.url:
        source_type = "url"
        source = args.url
    elif args.local:
        source_type = "local"
        source = args.local
    elif args.github:
        source_type = "github"
        source = args.github
    elif args.hf:
        source_type = "hf"
        source = args.hf
    else:
        raise ValueError("One of --url, --local, --github, or --hf must be provided")

    passphrase = get_passphrase(args.passphrase)
    chunk_size_bytes = args.chunk_size * 1024 * 1024

    tmpdir = tempfile.mkdtemp()
    try:
        src_path = sources.acquire(source_type, source, Path(tmpdir))
        session_id = str(uuid.uuid4())
        crypto.encrypt_stream(src_path, Path(tmpdir) / "encrypted", passphrase)
        chunks = splitter.split_file(Path(tmpdir) / "encrypted", session_id, Path(tmpdir), chunk_size_bytes)
        if args.drive:
            from innout import drive

            folder_url = drive.upload_to_drive(chunks, args.drive, args.credentials)
            print(f"Done. Session ID: {session_id}  Parts: {len(chunks)}")
            print(f"Drive folder: {folder_url}")
        else:
            try:
                uploader.upload_chunks(chunks, args.server, session_id, api_key=args.api_key)
            except uploader.MissingAPIKeyError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"Done. Session ID: {session_id}  Parts: {len(chunks)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def cmd_pull(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    passphrase = get_passphrase(args.passphrase)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "drive", None):
        from innout import drive
        tmpdir = tempfile.mkdtemp()
        try:
            chunks = drive.download_from_drive(
                args.drive, Path(tmpdir), getattr(args, "credentials", "credentials.json")
            )
            if not chunks:
                raise SystemExit(f"error: no files found in Drive folder {args.drive!r}")
            splitter.join_files(chunks, Path(tmpdir) / "encrypted")
            crypto.decrypt_stream(Path(tmpdir) / "encrypted", output_dir / "result", passphrase)
            print(f"Done. Output: {output_dir}/result")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    elif args.from_dir:
        from_dir = Path(args.from_dir)
        chunks = sorted(from_dir.glob("*.part???"))
        if not chunks:
            raise SystemExit(f"error: no chunk files (*.part???) found in {from_dir}")
        tmpdir = tempfile.mkdtemp()
        try:
            splitter.join_files(chunks, Path(tmpdir) / "encrypted")
            crypto.decrypt_stream(Path(tmpdir) / "encrypted", output_dir / "result", passphrase)
            print(f"Done. Output: {output_dir}/result")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        if not args.session_id:
            raise SystemExit("error: session_id is required when using --server")
        if not args.server:
            raise SystemExit("error: one of --server, --drive, or --from-dir is required")
        tmpdir = tempfile.mkdtemp()
        try:
            try:
                chunks = uploader.download_chunks(
                    args.server, args.session_id, Path(tmpdir), api_key=args.api_key
                )
            except uploader.MissingAPIKeyError as exc:
                raise SystemExit(str(exc)) from exc
            chunks = sorted(chunks, key=lambda p: p.name)
            splitter.join_files(chunks, Path(tmpdir) / "encrypted")
            crypto.decrypt_stream(
                Path(tmpdir) / "encrypted", output_dir / "result", passphrase
            )
            print(f"Done. Output: {output_dir}/result")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="innout",
        description="Encrypt, split, and upload large files/repos to a custom server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- push subcommand ---
    push_parser = subparsers.add_parser("push", help="Upload data to a server or Google Drive")
    source_group = push_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", metavar="<url>", help="Remote URL to fetch and push")
    source_group.add_argument("--local", metavar="<path>", help="Local path to push")
    source_group.add_argument("--github", metavar="owner/repo", help="GitHub repo to push")
    source_group.add_argument("--hf", metavar="org/model", help="HuggingFace model/dataset to push")
    dest_group = push_parser.add_mutually_exclusive_group()
    dest_group.add_argument("--server", metavar="<url>", help="innout server URL")
    dest_group.add_argument("--drive", metavar="<folder>", help="Google Drive folder name")
    push_parser.add_argument(
        "--credentials",
        metavar="<path>",
        default="credentials.json",
        help="OAuth credentials JSON for Google Drive (default: credentials.json)",
    )
    push_parser.add_argument("--passphrase", metavar="<str>", default=None, help="Encryption passphrase")
    push_parser.add_argument(
        "--api-key", metavar="<key>", default=None,
        help="Server API key (overrides INNOUT_API_KEY env var)"
    )
    push_parser.add_argument(
        "--chunk-size",
        metavar="<MB>",
        type=int,
        default=1800,
        help="Chunk size in MB (default: 1800)",
    )
    push_parser.set_defaults(func=cmd_push)

    # --- pull subcommand ---
    pull_parser = subparsers.add_parser("pull", help="Download and decrypt data")
    pull_parser.add_argument(
        "session_id", metavar="<session_id>", nargs="?", default=None,
        help="Session ID returned by push (required with --server)",
    )
    source_group_pull = pull_parser.add_mutually_exclusive_group()
    source_group_pull.add_argument("--server", metavar="<url>", help="innout server URL")
    source_group_pull.add_argument(
        "--drive", metavar="<folder>", help="Google Drive folder name to download from"
    )
    source_group_pull.add_argument(
        "--from-dir", metavar="<path>",
        help="Local directory containing manually downloaded chunk files",
    )
    pull_parser.add_argument(
        "--credentials",
        metavar="<path>",
        default="credentials.json",
        help="OAuth credentials JSON for Google Drive (default: credentials.json)",
    )
    pull_parser.add_argument("--passphrase", metavar="<str>", default=None, help="Decryption passphrase")
    pull_parser.add_argument(
        "--api-key", metavar="<key>", default=None,
        help="Server API key (overrides INNOUT_API_KEY env var)"
    )
    pull_parser.add_argument(
        "--output",
        metavar="<dir>",
        default=".",
        help="Output directory (default: current directory)",
    )
    pull_parser.set_defaults(func=cmd_pull)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
