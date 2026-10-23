"""
================================================================================
MODULE: Cryptographic Checksum & File Integrity Validator
FILE  : src/hygiene/O02_checksum_validator.py
PURPOSE:
  Provides enterprise-grade low-level file integrity verification before 
  payload ingestion. Executes file existence verification, zero-byte corruption 
  checks, and computes SHA-256 cryptographic hashes to prevent parsing partially 
  written, truncated, or tampered datasets.
================================================================================
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

# Configure module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoChecksumValidator")


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculates the SHA-256 cryptographic checksum of a file by reading in chunks.

    Args:
        file_path (Path): Path to the target file.
        chunk_size (int): Size of stream chunks to read in bytes (default 8KB).

    Returns:
        str: Hexadecimal SHA-256 checksum string.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_file_integrity(file_path: Path, expected_checksum: Optional[str] = None) -> bool:
    """
    Validates file existence, verifies it is not empty (zero-byte check), 
    and optionally validates against an expected SHA-256 checksum.

    Args:
        file_path (Path): Path to the target file.
        expected_checksum (str, optional): Expected SHA-256 checksum string.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    # 1. File Existence Check
    if not file_path.exists():
        logger.error(f"[INTEGRITY ERROR] Target file does not exist: {file_path}")
        return False

    # 2. Zero-Byte & Size Check
    file_size_bytes = file_path.stat().st_size
    logger.info(f"Inspecting file: {file_path.name} | Size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)")

    if file_size_bytes == 0:
        logger.critical(f"[CORRUPT PAYLOAD] Zero-byte check FAILED: {file_path.name} is empty (0 bytes)!")
        return False

    logger.info(f"Zero-byte check PASSED for: {file_path.name}")

    # 3. Cryptographic Checksum Computation
    actual_checksum = calculate_sha256(file_path)
    logger.info(f"Computed SHA-256 Checksum: {actual_checksum}")

    if expected_checksum:
        if actual_checksum.lower() == expected_checksum.lower():
            logger.info("Checksum verification PASSED! File signature matches expected hash.")
            return True
        else:
            logger.critical(
                f"[CHECKSUM MISMATCH] Expected '{expected_checksum}' but found '{actual_checksum}'. "
                "File may be corrupt, modified, or truncated!"
            )
            return False

    return True


if __name__ == "__main__":
    test_path = Path("../../data/raw/Telco-Customer-Churn.csv")
    if test_path.exists():
        validate_file_integrity(test_path)
    else:
        print(f"File not found at {test_path}")
