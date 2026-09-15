from pathlib import Path

import pymupdf  # PyMuPDF
import requests
import re

PDF_URL = (
    "https://www.uscis.gov/sites/default/files/document/"
    "policy-manual-updates/20240827-STEMOPT.pdf"
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PDF_PATH = RAW_DIR / "uscis_stem_opt_policy_alert.pdf"
TEXT_PATH = PROCESSED_DIR / "uscis_stem_opt_policy_alert.txt"


def download_pdf(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Visa-GPT research project"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Downloaded PDF: {output_path}")

def clean_text(text: str) -> str:
    # Remove trailing spaces on lines
    lines = [line.strip() for line in text.splitlines()]

    text = "\n".join(lines)

    # Replace 3+ consecutive blank lines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    document.close()

    return "\n".join(pages)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    download_pdf(PDF_URL, PDF_PATH)

    text = extract_pdf_text(PDF_PATH)
    text = clean_text(text)

    TEXT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Extracted text: {TEXT_PATH}")
    print(f"Characters extracted: {len(text):,}")


if __name__ == "__main__":
    main()