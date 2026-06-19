"""
SEC EDGAR 10-K Downloader
Downloads the most recent 10-K filing for a given company CIK.
"""

import os
import re
import time
import requests
from pathlib import Path

import config


def _get_headers() -> dict:
    """SEC requires a User-Agent header for all programmatic access."""
    return {
        "User-Agent": config.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def _pad_cik(cik: str) -> str:
    """Pad CIK to 10 digits as required by SEC API."""
    return cik.lstrip("0").zfill(10)


def get_latest_10k_url(cik: str) -> tuple[str, str]:
    """
    Query SEC EDGAR API to find the most recent 10-K filing URL.
    
    Returns:
        Tuple of (filing_url, accession_number)
    """
    padded_cik = _pad_cik(cik)
    submissions_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    
    print(f"  Querying SEC EDGAR for CIK {padded_cik}...")
    response = requests.get(submissions_url, headers=_get_headers())
    response.raise_for_status()
    
    data = response.json()
    company_name = data.get("name", "Unknown")
    print(f"  Company: {company_name}")
    
    # Search recent filings for 10-K
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i]
            primary_doc = primary_docs[i]
            # Build the filing URL
            accession_no_dashes = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{padded_cik}/{accession_no_dashes}/{primary_doc}"
            )
            print(f"  Found 10-K: accession {accession}")
            return filing_url, accession
    
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def download_10k(cik: str, force: bool = False) -> Path:
    """
    Download the most recent 10-K filing for the given CIK.
    Caches to data/ directory to avoid re-downloading.
    
    Args:
        cik: SEC CIK number (e.g. "0001045810" for NVIDIA)
        force: If True, re-download even if cached
        
    Returns:
        Path to the downloaded HTML file
    """
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache first
    cached_files = list(data_dir.glob(f"10k_{cik}_*.htm*"))
    if cached_files and not force:
        print(f"  Using cached file: {cached_files[0]}")
        return cached_files[0]
    
    # Download
    filing_url, accession = get_latest_10k_url(cik)
    
    # Respect SEC rate limit (10 req/sec max)
    time.sleep(0.2)
    
    print(f"  Downloading from: {filing_url}")
    response = requests.get(filing_url, headers=_get_headers())
    response.raise_for_status()
    
    # Determine file extension from URL
    ext = ".htm" if filing_url.endswith(".htm") else ".html"
    filename = f"10k_{cik}_{accession}{ext}"
    filepath = data_dir / filename
    
    filepath.write_text(response.text, encoding="utf-8")
    print(f"  Saved to: {filepath} ({len(response.text):,} chars)")
    
    return filepath
