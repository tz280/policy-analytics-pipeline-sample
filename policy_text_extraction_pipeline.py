"""
Policy Document Text Extraction Pipeline
=========================================
Author: Tongrui (Neil) Zhang
Project: Cross-Country Investment Policy Analysis
         Georgetown University McCourt School of Public Policy

Description:
    This pipeline processes OCR-extracted policy documents from 110+ countries,
    extracting structured topic-level text segments for downstream NLP classification
    and econometric modeling. The pipeline handles common OCR artifacts including
    character spacing errors, table noise, and table-of-contents contamination.

Input:
    - Directory of plain-text (.txt) files, each named as {country}_{year}.txt
    - Files are OCR-extracted from official government policy documents

Output:
    - Structured CSV with one row per country-year, columns per policy topic
    - Each cell contains the raw extracted text for that topic in that document

Pipeline Stages:
    1. Document ingestion and filename parsing
    2. Section structure extraction (heading-based segmentation)
    3. Table-of-contents filtering
    4. Table and noise removal
    5. Regex-based topic matching with OCR-tolerance patterns
    6. Special-case topic extraction (e.g., media sentiment subsections)
    7. Batch processing and CSV export

Usage:
    python policy_text_extraction_pipeline.py
    
    Update INPUT_DIR and OUTPUT_CSV in the Configuration section before running.
"""

import os
import re
import pandas as pd


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# --- Topic Definitions -------------------------------------------------------
# Topics to extract and include as columns in the output CSV.
# These map to section headings found in the source policy documents.

TOPIC_COLUMNS = [
    "commercial_bribery",
    "media_attitude_toward_china",
    "media_relations",
    "labor_employment_regulations",
    "law_enforcement_relations",
    "foreign_contracting",
    "trade_union_relations",
    "union_relationship_management",
    "overseas_contracting",
    "overseas_labor_cooperation",
    "environmental_regulations",
    "ecological_protection",
    "labor_supply_and_wages",
]

# --- Topic Matching Patterns -------------------------------------------------
# Regex patterns are OCR-tolerant: \s* allows zero or more whitespace between
# characters to handle common OCR spacing errors in Chinese text.

TOPIC_PATTERNS = {
    "commercial_bribery": re.compile(r"商\s*业\s*贿\s*赂"),
    "media_relations": re.compile(r"(?:懂\s*得)?(?:与|和)\s*媒\s*体\s*打\s*交\s*道"),
    "labor_employment_regulations": re.compile(r"劳\s*动\s*就\s*业.*(?:法\s*规|规\s*定)"),
    "law_enforcement_relations": re.compile(r"(?:与|和)\s*执\s*法\s*人\s*员\s*打\s*交\s*道"),
    "foreign_contracting": re.compile(r"承\s*包\s*当\s*地\s*工\s*程"),
    "trade_union_relations": re.compile(r"工\s*会(?:及|和)(?:其\s*他)?(?:非\s*政\s*府)?\s*组\s*织"),
    "union_relationship_management": re.compile(r"(?:妥\s*善\s*处\s*理)?(?:与|和)?\s*工\s*会\s*的\s*关\s*系"),
    "overseas_contracting": re.compile(r"(?:\d+\.\d+\s*)?对\s*外\s*承\s*包\s*工\s*程"),
    "overseas_labor_cooperation": re.compile(r"(?:\d+\.\d+\s*)?对\s*外\s*劳\s*务\s*合\s*作"),
    "environmental_regulations": re.compile(r"环\s*境\s*(?:保\s*护|法\s*律).*法\s*规"),
    "ecological_protection": re.compile(r"依\s*法\s*保\s*护(?:生\s*态|当\s*地)[环坏]\s*境"),
    "labor_supply_and_wages": re.compile(
        r"劳\s*动\s*力(?:"
        r"(?:供\s*求|供\s*应|供\s*需).*?(?:工\s*薪|薪\s*酬)|"
        r".*?(?:工\s*薪|薪\s*酬).*?(?:供\s*求|供\s*应|供\s*需)"
        r")"
    ),
}

# --- Special Topic Definitions -----------------------------------------------
# Some topics require sub-section extraction within a matched heading.

SPECIAL_TOPICS = {
    "media_attitude_toward_china": {
        "section_pattern": re.compile(r"主\s*要\s*媒\s*体"),
    }
}

# --- Structural Patterns -----------------------------------------------------
# Matches numbered headings of the form: 3.5, 3.5.1, 1.2.3.4, etc.
HEADING_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)+)\s*([^\n]+)\s*$")

# --- I/O Configuration -------------------------------------------------------
INPUT_DIR = "data/raw_policy_texts"     # Directory of {country}_{year}.txt files
OUTPUT_CSV = "outputs/policy_topics_extracted.csv"
VALID_YEARS = {"2022", "2023"}          # Filter: only process files for these years


# =============================================================================
# SECTION 2: DOCUMENT STRUCTURE EXTRACTION
# =============================================================================

def is_toc_line(heading: str) -> bool:
    """
    Detect whether a heading line belongs to the table of contents (TOC)
    rather than the document body.

    TOC lines typically contain:
        - Sequences of dots used as fill characters (e.g., ".........")
        - Trailing page numbers
        - Repetitive whitespace-dot patterns

    Args:
        heading: Candidate heading string.

    Returns:
        True if the line is identified as a TOC entry; False otherwise.
    """
    if re.search(r'\.{3,}', heading):
        return True
    if re.search(r'\s+\d+\s*$', heading):
        return True
    if heading.count('.') > 5:
        return True
    if re.search(r'[\s\.]{10,}', heading):
        return True
    return False


def extract_sections(text: str) -> list[dict]:
    """
    Segment a document into structured sections based on numbered headings.

    Each section spans from its own heading to the next heading at the same
    hierarchical level. TOC lines are excluded.

    Args:
        text: Full document text.

    Returns:
        List of dicts, each containing:
            - number  (str): Heading number, e.g., "3.5.1"
            - heading (str): Heading text
            - level   (int): Depth level (1 = top, higher = deeper)
            - section (str): Full text of the section including heading
    """
    matches = list(HEADING_RE.finditer(text))
    sections = []

    for i, match in enumerate(matches):
        number = match.group(1)
        heading = match.group(2).strip()

        if is_toc_line(heading):
            continue

        level = number.count(".") + 1
        start = match.start()

        # Section ends at the next heading at the same depth level
        end = len(text)
        for j in range(i + 1, len(matches)):
            next_number = matches[j].group(1)
            if next_number.count(".") + 1 == level:
                end = matches[j].start()
                break

        sections.append({
            "number": number,
            "heading": heading,
            "level": level,
            "section": text[start:end].strip(),
        })

    return sections


# =============================================================================
# SECTION 3: TEXT CLEANING
# =============================================================================

def remove_tables(text: str) -> str:
    """
    Remove embedded table blocks from section text.

    Targets patterns of the form:
        表N-N：<title>
        <table content>
        资料来源/数据来源：<source>

    Args:
        text: Raw section text.

    Returns:
        Text with table blocks removed and excess blank lines collapsed.
    """
    table_pattern = re.compile(
        r'表\d+-\d+[：:][^\n]*\n.*?(?:资料来源|数据来源)[：:][^\n]*',
        re.DOTALL
    )
    cleaned = table_pattern.sub('', text)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


def remove_toc_residuals(text: str, country_list: list[str]) -> str:
    """
    Remove leftover TOC fragments that appear as isolated country names
    followed by page numbers (a common OCR artifact when TOC spans multiple pages).

    Args:
        text: Section text that may contain TOC residuals.
        country_list: List of known country name strings to match against.

    Returns:
        Cleaned text with TOC residual lines removed.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Drop lines consisting only of digits/punctuation (standalone page numbers)
        if re.match(r'^[\d\s\.,\-]+$', line):
            continue
        # Drop short lines that are a country name followed by a page number
        if any(country in line for country in country_list):
            if re.search(r'\d+', line) and len(line) < 30:
                continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


# =============================================================================
# SECTION 4: TOPIC MATCHING
# =============================================================================

def extract_media_attitude_subsection(section_text: str) -> str:
    """
    Extract the sub-section discussing media attitude toward China from within
    a broader 'major media' section.

    Searches for keyword patterns indicating the start of the relevant sub-section,
    then returns all text from that point onward. Handles bracketed annotations
    that may precede the actual content.

    Args:
        section_text: Full text of the 'major media' section.

    Returns:
        Extracted sub-section text, or empty string if not found.
    """
    keyword_patterns = [
        re.compile(r"对\s*华\s*舆\s*情"),
        re.compile(r"对\s*华\s*舆\s*论"),
        re.compile(r"对\s*中\s*舆\s*情"),
        re.compile(r"对\s*中\s*舆\s*论"),
        re.compile(r"媒\s*体\s*对\s*中\s*态\s*度"),
        re.compile(r"媒\s*体\s*对\s*华\s*态\s*度"),
        re.compile(r"对\s*华\s*态\s*度"),
        re.compile(r"对\s*中\s*态\s*度"),
    ]

    for pattern in keyword_patterns:
        match = pattern.search(section_text)
        if match:
            pos = match.start()
            # If match falls inside a 【...】 annotation, advance past the closing bracket
            context = section_text[max(0, pos - 10): pos + 20]
            if '【' in context and '】' in context:
                bracket_end = section_text.find('】', pos)
                if bracket_end != -1:
                    pos = bracket_end + 1
            return section_text[pos:].strip()

    return ""


def assign_topics(text: str, country_list: list[str]) -> dict:
    """
    Match all defined topics against the sections of a document and extract
    the corresponding text for each.

    Processing order:
        1. Extract all sections from the document.
        2. For each section, clean table noise and TOC residuals.
        3. Apply regex patterns to match section headings to topics.
        4. Apply special-case extraction for sub-section topics.

    Args:
        text: Full document text.
        country_list: Country name list passed through for TOC residual removal.

    Returns:
        Dict mapping each topic column name to its extracted text (empty str if not found).
    """
    sections = extract_sections(text)
    result = {topic: "" for topic in TOPIC_COLUMNS}

    for sec in sections:
        heading = sec["heading"]
        section_text = sec["section"]

        # Apply cleaning
        cleaned = remove_tables(section_text)
        cleaned = remove_toc_residuals(cleaned, country_list)

        # Standard topic matching: first match wins to avoid partial overwrites
        for topic, pattern in TOPIC_PATTERNS.items():
            if pattern.search(heading) and not result[topic]:
                result[topic] = cleaned

        # Special case: extract media-attitude sub-section
        special_cfg = SPECIAL_TOPICS["media_attitude_toward_china"]
        if special_cfg["section_pattern"].search(heading):
            result["media_attitude_toward_china"] = extract_media_attitude_subsection(cleaned)

    return result


# =============================================================================
# SECTION 5: FILE-LEVEL PROCESSING
# =============================================================================

def parse_filename(filename: str) -> tuple[str, str] | None:
    """
    Parse country name and year from a filename of the form {country}_{year}.txt.

    Args:
        filename: File name string (with or without path).

    Returns:
        (country, year) tuple, or None if the filename does not match expected format.
    """
    name = os.path.basename(filename).replace('.txt', '')
    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def process_file(file_path: str, filename: str, country_list: list[str]) -> dict | None:
    """
    Process a single policy document file.

    Args:
        file_path: Absolute path to the .txt file.
        filename: Base filename used for metadata parsing.
        country_list: Country name list for TOC residual removal.

    Returns:
        Dict with 'country', 'year', and one key per topic column,
        or None if the file cannot be parsed.
    """
    parsed = parse_filename(filename)
    if parsed is None:
        print(f"  [SKIP] Unexpected filename format: {filename}")
        return None

    country, year = parsed

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except OSError as e:
        print(f"  [ERROR] Could not read {filename}: {e}")
        return None

    topic_data = assign_topics(content, country_list)

    return {"country": country, "year": year, **topic_data}


# =============================================================================
# SECTION 6: BATCH PROCESSING
# =============================================================================

# Known country name strings used for TOC residual filtering.
# Extend this list to improve cleaning coverage.
KNOWN_COUNTRIES = [
    "东盟", "丹麦", "乌克兰", "乌兹别克", "亚美尼亚", "以色列", "伊拉克", "伊朗",
    "冰岛", "刚果共和国", "利比亚", "加拿大", "加蓬", "匈牙利", "南苏丹", "南非",
    "印度尼西亚", "哈萨克斯坦", "哥伦比亚", "土库曼斯坦", "坦桑尼亚", "塔吉克斯坦",
    "墨西哥", "奥地利", "安哥拉", "巴基斯坦", "巴西", "德国", "斐济", "斯里兰卡",
    "新西兰", "日本", "智利", "朝鲜", "格鲁吉亚", "比利时", "泰国", "洪都拉斯",
    "爱尔兰", "爱沙尼亚", "白俄罗斯", "科威特", "秘鲁", "缅甸", "美国", "老挝",
    "芬兰", "苏丹", "莫桑比克", "蒙古", "西班牙", "阿根廷", "马来西亚", "黎巴嫩",
]


def run_pipeline(
    input_dir: str = INPUT_DIR,
    output_csv: str = OUTPUT_CSV,
    valid_years: set[str] = VALID_YEARS,
) -> None:
    """
    Main entry point: batch-process all policy documents and export results.

    Args:
        input_dir:   Directory containing {country}_{year}.txt files.
        output_csv:  Path for the output CSV file.
        valid_years: Set of year strings to include (others are skipped).
    """
    print("=" * 70)
    print("Policy Document Topic Extraction Pipeline")
    print("=" * 70)

    if not os.path.isdir(input_dir):
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        return

    # Collect and filter target files
    all_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.txt') and any(yr in f for yr in valid_years)
    ]
    all_files.sort()
    print(f"\nFound {len(all_files)} files to process.\n")
    print("-" * 70)

    # Process files
    rows = []
    for filename in all_files:
        file_path = os.path.join(input_dir, filename)
        row = process_file(file_path, filename, KNOWN_COUNTRIES)
        if row:
            rows.append(row)
            print(f"  [OK]  {row['country']:<30s}  {row['year']}")
        else:
            print(f"  [SKIP] {filename}")

    # Export results
    print("\n" + "=" * 70)
    if not rows:
        print("[WARNING] No data extracted. Check input directory and filename format.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["country", "year"] + TOPIC_COLUMNS)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Pipeline complete.")
    print(f"  Rows exported : {len(df)}")
    print(f"  Columns       : {len(df.columns)}  (country + year + {len(TOPIC_COLUMNS)} topics)")
    print(f"  Output file   : {output_csv}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()
