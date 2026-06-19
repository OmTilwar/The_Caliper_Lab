"""
10-K HTML Parser
Extracts semantic sections from SEC 10-K filings and converts them to clean text.
"""

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from src.models import Section


# SEC 10-K standard items we want to extract
# Maps regex pattern -> human-readable section name
SECTION_PATTERNS = [
    (r"item\s*1a[.\s:]*risk\s*factors", "Item 1A: Risk Factors"),
    (r"item\s*1b[.\s:]*unresolved\s*staff", "Item 1B: Unresolved Staff Comments"),
    (r"item\s*1[.\s:]*business", "Item 1: Business"),
    (r"item\s*7a[.\s:]*quantitative", "Item 7A: Market Risk Disclosures"),
    (r"item\s*7[.\s:]*management", "Item 7: MD&A"),
    (r"item\s*8[.\s:]*financial\s*statements", "Item 8: Financial Statements"),
]

# Sections we actually want to generate QA from (the most content-rich ones)
TARGET_SECTIONS = {
    "Item 1: Business",
    "Item 1A: Risk Factors",
    "Item 7: MD&A",
    "Item 7A: Market Risk Disclosures",
    "Item 8: Financial Statements",
}


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    # Replace multiple newlines with double newline
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Replace multiple spaces with single space (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove page numbers and common artifacts
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Remove common 10-K boilerplate lines
    text = re.sub(r"Table of Contents\s*", "", text, flags=re.IGNORECASE)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_markdown(table: Tag) -> str:
    """Convert an HTML table to a markdown-formatted table."""
    rows = table.find_all("tr")
    if not rows:
        return ""
    
    md_rows = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        cell_texts = []
        for cell in cells:
            # Get text, clean whitespace
            cell_text = cell.get_text(separator=" ", strip=True)
            cell_text = re.sub(r"\s+", " ", cell_text)
            # Handle colspan
            colspan = int(cell.get("colspan", 1))
            cell_texts.append(cell_text)
            # Add empty cells for colspan
            for _ in range(colspan - 1):
                cell_texts.append("")
        
        if any(cell_texts):  # Skip completely empty rows
            md_rows.append("| " + " | ".join(cell_texts) + " |")
    
    if len(md_rows) < 2:
        return "\n".join(md_rows)
    
    # Add separator after first row (header)
    num_cols = md_rows[0].count("|") - 1
    separator = "|" + "|".join(["---"] * num_cols) + "|"
    md_rows.insert(1, separator)
    
    return "\n".join(md_rows)


def _extract_text_with_tables(element: Tag) -> str:
    """
    Extract text from an HTML element, converting tables to markdown format.
    This preserves numeric data in tables which is critical for calculation questions.
    """
    parts = []
    
    for child in element.children:
        if isinstance(child, Tag):
            if child.name == "table":
                md_table = _table_to_markdown(child)
                if md_table:
                    parts.append("\n\n" + md_table + "\n\n")
            elif child.name in ("script", "style"):
                continue
            else:
                parts.append(_extract_text_with_tables(child))
        elif isinstance(child, str):
            text = child.strip()
            if text:
                parts.append(text + " ")
    
    return "".join(parts)


def _find_section_boundaries(soup: BeautifulSoup) -> list[tuple[str, int, Tag]]:
    """
    Find section boundaries by searching for Item headers in the document.
    Returns list of (section_name, position, element) tuples.
    """
    boundaries = []
    
    # Get all text-containing elements
    all_elements = soup.find_all(["p", "div", "span", "b", "strong", "h1", "h2", "h3", "h4", "font", "a"])
    
    for i, elem in enumerate(all_elements):
        text = elem.get_text(strip=True)
        if not text or len(text) > 200:  # Skip very long text (not a header)
            continue
        
        text_lower = text.lower()
        
        for pattern, section_name in SECTION_PATTERNS:
            if re.search(pattern, text_lower):
                # Avoid matching TOC entries (they tend to be in <a> tags with href)
                # and tend to be very short with just the item reference
                parent = elem.parent
                is_toc = False
                if elem.name == "a" and elem.get("href"):
                    is_toc = True
                # Also check if this is inside a TOC-like structure
                if parent and parent.get_text(strip=True) == text:
                    # The parent only contains this text - likely a real header
                    pass
                    
                if not is_toc:
                    boundaries.append((section_name, i, elem))
                    break
    
    return boundaries


def parse_10k(filepath: Path) -> list[Section]:
    """
    Parse a 10-K HTML filing into semantic sections.
    
    Args:
        filepath: Path to the downloaded 10-K HTML file
        
    Returns:
        List of Section objects for the target sections
    """
    print(f"  Parsing {filepath.name}...")
    
    html_content = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Strategy: extract all text with structure, then split by section headers
    # This is more robust than trying to navigate the DOM tree
    
    # First, try to get the full document text with tables preserved
    body = soup.find("body") or soup
    full_text = _extract_text_with_tables(body)
    
    # Find section boundaries in the text
    sections = []
    
    # Build regex patterns for finding sections in the extracted text
    section_markers = []
    for pattern, section_name in SECTION_PATTERNS:
        # Find all matches in the full text
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            section_markers.append((match.start(), section_name))
    
    # Sort by position
    section_markers.sort(key=lambda x: x[0])
    
    # Deduplicate: keep only the first occurrence that looks like a real header
    # (skip TOC entries which appear earlier in the document)
    seen_sections = {}
    for pos, name in section_markers:
        if name not in seen_sections:
            seen_sections[name] = []
        seen_sections[name].append(pos)
    
    # For each section, use the LAST occurrence (TOC entries come first, 
    # actual content headers come later)
    final_markers = []
    for name, positions in seen_sections.items():
        if name in TARGET_SECTIONS:
            # Use the last occurrence if there are multiple
            # (first is usually TOC, last is the actual section)
            if len(positions) > 1:
                final_markers.append((positions[-1], name))
            else:
                final_markers.append((positions[0], name))
    
    final_markers.sort(key=lambda x: x[0])
    
    # Extract text between section markers
    for i, (start_pos, section_name) in enumerate(final_markers):
        if i + 1 < len(final_markers):
            end_pos = final_markers[i + 1][0]
        else:
            end_pos = len(full_text)
        
        section_text = full_text[start_pos:end_pos]
        section_text = _clean_text(section_text)
        
        if len(section_text) > 500:  # Skip very short sections (likely misdetected)
            sections.append(Section(
                name=section_name,
                full_text=section_text,
            ))
            print(f"    ✓ {section_name}: {len(section_text):,} chars")
    
    if not sections:
        # Fallback: if section detection fails, treat the whole document as one section
        print("  ⚠ Section detection failed, using full document as single section")
        full_cleaned = _clean_text(full_text)
        # Split into roughly equal parts
        chunk_size = len(full_cleaned) // 5
        for i in range(5):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 4 else len(full_cleaned)
            section_text = full_cleaned[start:end]
            sections.append(Section(
                name=f"Section {i+1}",
                full_text=section_text,
            ))
    
    print(f"  Extracted {len(sections)} sections")
    return sections
