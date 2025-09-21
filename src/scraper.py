import os
import csv
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import mimetypes
from playwright.sync_api import sync_playwright

BASE_URL = "https://aiap.sg/apprenticeship/"
OUTPUT_DIR = "corpus"
VISITED = set()
REPORT = []

# ensure output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid(url):
    parsed = urlparse(url)
    return parsed.netloc == urlparse(BASE_URL).netloc

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

def save_file(url, page_url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        ext = mimetypes.guess_extension(content_type.split(';')[0]) or os.path.splitext(url)[1]
        fname = sanitize_filename(os.path.basename(urlparse(url).path) or "file") + ext
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "wb") as f:
            f.write(resp.content)
        REPORT.append((page_url, url, fname))
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def convert_page_to_pdf(url,counter=0):
    url_parse = urlparse(url)
    fname = url_parse.path if url_parse.path else f"file-{counter}"
    fname += ".pdf"
    fname = sanitize_filename(fname.replace("/", ""))
    path = os.path.join(OUTPUT_DIR, fname)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')
        page.pdf(path=path, format="A4")
        browser.close()

def save_code_blocks(soup, page_url):
    code_blocks = soup.find_all(["code", "pre"])
    for i, block in enumerate(code_blocks):
        code_text = block.get_text().strip()
        if code_text:
            fname = sanitize_filename(f"{urlparse(page_url).path.strip('/').replace('/', '_')}_code_{i+1}.txt")
            path = os.path.join(OUTPUT_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write("# CODE BLOCK\n")
                f.write(code_text)
            REPORT.append((page_url, None, fname))

def process_page(url):
    if url in VISITED:
        return
    if urlparse(url).fragment:
        # Avoid duplicate
        VISITED.add(url)
        return
    VISITED.add(url)
    print(f"Scraping page: {url}")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        html_content = resp.text
    except Exception as e:
        print(f"Failed to fetch page {url}: {e}")
        return

    soup = BeautifulSoup(html_content, "html.parser")

    # Convert HTML to PDF
    convert_page_to_pdf(url, counter=len(VISITED))

    # Extract and save code blocks
    save_code_blocks(soup, url)

    # Download linked assets
    for tag in soup.find_all(["a", "img"]):
        src = tag.get("href") or tag.get("src")
        if not src:
            continue
        asset_url = urljoin(url, src)
        if not is_valid(asset_url):
            continue
        lower = asset_url.lower()
        if any(lower.endswith(ext) for ext in [".pdf", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".gif"]):
            save_file(asset_url, url)

    # Recurse through internal links
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        if is_valid(link):
            process_page(link)

def write_report():
    report_path = os.path.join(OUTPUT_DIR, "scrape_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_url", "asset_url", "local_filename"])
        for row in REPORT:
            writer.writerow(row)
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    process_page(BASE_URL)
    write_report()
