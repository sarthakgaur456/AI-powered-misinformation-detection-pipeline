import requests
from bs4 import BeautifulSoup
import csv
import time

BASE = "https://www.politifact.com"
INDEX_URL = "https://www.politifact.com/factchecks/?page={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; simple-scraper/1.0)"}

def scrape_detail(url):
    """Scrape full details from a single Politifact fact-check page."""
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    claim = soup.select_one("div.m-statement__quote")
    claim_text = claim.get_text(strip=True) if claim else ""

    speaker = soup.select_one("a.m-statement__name")
    speaker_name = speaker.get_text(strip=True) if speaker else ""

    desc = soup.select_one("div.m-statement__desc")
    desc_text = desc.get_text(strip=True) if desc else ""

    ruling_img = soup.select_one("div.m-statement__meter img")
    ruling = ruling_img["alt"] if ruling_img and "alt" in ruling_img.attrs else ""

    tags = [tag.get_text(strip=True) for tag in soup.select("ul.m-list--horizontal li span")]

    author = soup.select_one("div.m-author__content a")
    author_name = author.get_text(strip=True) if author else ""

    date = soup.select_one("span.m-author__date")
    date_text = date.get_text(strip=True) if date else ""

    summary_bullets = [li.get_text(strip=True) for li in soup.select("div.short-on-time li")]

    article_paras = [p.get_text(strip=True) for p in soup.select("article.m-textblock p")]

    sources = [a.get("href") for a in soup.select("section#sources a")]

    return {
        "url": url,
        "claim": claim_text,
        "speaker": speaker_name,
        "desc": desc_text,
        "ruling": ruling,
        "tags": "; ".join(tags),
        "author": author_name,
        "date": date_text,
        "summary": " | ".join(summary_bullets),
        "article": " ".join(article_paras),
        "sources": "; ".join(sources)
    }

def scrape_politifact(pages=2, delay=1.0):
    """Scrape Politifact index pages and then each claim detail."""
    all_data = []

    for page in range(1, pages + 1):
        print(f"Fetching index page {page}...")
        r = requests.get(INDEX_URL.format(page), headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        items = soup.select("li.o-listicle__item article.m-statement")
        for it in items:
            link_tag = it.select_one(".m-statement__quote a")
            if not link_tag:
                continue
            link = BASE + link_tag["href"]
            print(f"  Scraping: {link}")

            try:
                data = scrape_detail(link)
                all_data.append(data)
            except Exception as e:
                print("   Error:", e)

            time.sleep(delay)  # polite delay between requests

    return all_data

if __name__ == "__main__":
    data = scrape_politifact(pages=50, delay=1)  # increase pages for more data

    with open("politifact_full_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url","claim","speaker","desc","ruling","tags",
                "author","date","summary","article","sources"
            ]
        )
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} records to politifact_full_dataset.csv")
