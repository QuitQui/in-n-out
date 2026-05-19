from pathlib import Path

from tqdm import tqdm

DEFAULT_CHUNK_SIZE = 1_800 * 1024 * 1024  # 1.8 GB

_READ_BUFFER = 4 * 1024 * 1024  # 4 MB


def split_file(
    src: Path,
    session_id: str,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[Path]:
    """Split src into chunks of at most chunk_size bytes.

    Each chunk is written to output_dir/<session_id>.part<NN> (zero-padded to 3 digits).
    Returns the list of chunk paths in order.
    Shows a tqdm progress bar based on total bytes.
    """
    total_size = src.stat().st_size
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    chunk_index = 0

    with src.open("rb") as src_file:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Splitting {src.name}") as progress:
            while True:
                chunk_path = output_dir / f"{session_id}.part{chunk_index:03d}"
                bytes_written = 0

                with chunk_path.open("wb") as chunk_file:
                    while bytes_written < chunk_size:
                        read_size = min(_READ_BUFFER, chunk_size - bytes_written)
                        data = src_file.read(read_size)
                        if not data:
                            break
                        chunk_file.write(data)
                        bytes_written += len(data)
                        progress.update(len(data))

                if bytes_written == 0:
                    # No data was written; remove the empty file and stop
                    chunk_path.unlink()
                    break

                chunk_paths.append(chunk_path)
                chunk_index += 1

                if bytes_written < chunk_size:
                    # Reached end of source file
                    break

    return chunk_paths


def join_files(chunks: list[Path], output_path: Path) -> None:
    """Concatenate chunks (in order) into output_path.

    Shows a tqdm progress bar.
    chunks must already be sorted in the correct order.
    """
    total_size = sum(chunk.stat().st_size for chunk in chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as out_file:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Joining {output_path.name}") as progress:
            for chunk in chunks:
                with chunk.open("rb") as chunk_file:
                    while True:
                        data = chunk_file.read(_READ_BUFFER)
                        if not data:
                            break
                        out_file.write(data)
                        progress.update(len(data))
