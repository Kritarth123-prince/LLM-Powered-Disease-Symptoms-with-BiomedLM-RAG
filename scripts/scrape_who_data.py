import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.who.int"
FACTSHEET_LIST_URL = f"{BASE_URL}/news-room/fact-sheets/"

class WHOScraper:
    """Scraper for WHO fact sheets"""
    
    def __init__(self, output_path: str = "data/raw/who_dataset.json"):
        self.base_url = BASE_URL
        self.factsheet_url = FACTSHEET_LIST_URL
        self.output_path = Path(output_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_fact_sheet_links(self) -> List[str]:
        """Fetch all WHO fact sheet links from the main page."""
        logger.info(f"Fetching fact sheet links from {self.factsheet_url}")
        
        try:
            response = self.session.get(self.factsheet_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch fact sheet list: {e}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news-room/fact-sheets/detail/" in href:
                full_url = href if href.startswith("http") else self.base_url + href
                if full_url not in links:
                    links.append(full_url)
        
        logger.info(f"Found {len(links)} fact sheet links")
        return links
    
    def extract_fact_sheet(self, url: str) -> Dict[str, Any]:
        """Extracts details from a single WHO fact sheet."""
        data = {
            "name": "",
            "url": url,
            "key_facts": "",
            "overview": "",
            "impact": "",
            "symptoms": "",
            "causes": "",
            "treatment": "",
            "self_care": "",
            "who_response": "",
            "reference": ""
        }
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return data
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract title
        title_tag = soup.find("h1")
        if title_tag:
            data["name"] = title_tag.get_text(strip=True)
        
        # Extract Key Facts
        key_facts_heading = soup.find(
            lambda tag: tag.name in ["h2", "h3"] and 
            "key facts" in tag.get_text(strip=True).lower()
        )
        if key_facts_heading:
            ul = key_facts_heading.find_next("ul")
            if ul:
                key_facts = [li.get_text(strip=True) for li in ul.find_all("li")]
                data["key_facts"] = "\n".join(key_facts)
        
        # Extract sections from body
        body = soup.find("div", class_=lambda c: c and "sf-detail-body-wrapper" in c)
        if not body:
            body = soup
        
        sections = {
            "overview": ["overview"],
            "impact": ["impact"],
            "symptoms": ["symptoms", "signs and symptoms"],
            "causes": ["causes", "cause"],
            "treatment": ["treatment", "treatments"],
            "self_care": ["self-care", "self care", "prevention"],
            "who_response": ["who response", "who's response", "who's work", "who activities"],
            "reference": ["reference", "references"]
        }
        
        headings = body.find_all(["h2", "h3"])
        for heading in headings:
            title = heading.get_text(strip=True).lower()
            for key, variants in sections.items():
                if any(v in title for v in variants):
                    content = []
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ["h2", "h3"]:
                            break
                        if sibling.name == "p":
                            text = sibling.get_text(strip=True)
                            if text:
                                content.append(text)
                        elif sibling.name == "ul":
                            for li in sibling.find_all("li"):
                                text = li.get_text(strip=True)
                                if text:
                                    content.append(text)
                    if content:
                        data[key] = "\n".join(content)
        
        return data
    
    def scrape_all(self, delay: float = 1.0, max_pages: int = None):
        """Scrape all fact sheets"""
        logger.info("Starting WHO fact sheets scraping...")
        
        # Get all links
        fact_sheet_links = self.get_fact_sheet_links()
        
        if max_pages:
            fact_sheet_links = fact_sheet_links[:max_pages]
            logger.info(f"Limited to {max_pages} pages for testing")
        
        all_data = []
        failed_urls = []
        
        for i, url in enumerate(fact_sheet_links, start=1):
            logger.info(f"[{i}/{len(fact_sheet_links)}] Extracting: {url}")
            
            try:
                fact_data = self.extract_fact_sheet(url)
                if fact_data["name"]:  # Only add if we got a title
                    all_data.append(fact_data)
                    logger.info(f"✓ Successfully extracted: {fact_data['name']}")
                else:
                    logger.warning(f"✗ No data extracted from {url}")
                    failed_urls.append(url)
            except Exception as e:
                logger.error(f"✗ Error extracting {url}: {e}")
                failed_urls.append(url)
            
            # Polite delay
            time.sleep(delay)
        
        # Save results
        self.save_data(all_data)
        
        # Save failed URLs
        if failed_urls:
            failed_path = self.output_path.parent / "failed_urls.json"
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(failed_urls, f, indent=2)
            logger.warning(f"Failed to scrape {len(failed_urls)} URLs. Saved to {failed_path}")
        
        logger.info(f"Scraping complete! Successfully scraped {len(all_data)} fact sheets")
        return all_data
    
    def save_data(self, data: List[Dict[str, Any]]):
        """Save scraped data to JSON file"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {self.output_path}")


def main():
    """Main function to run the scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape WHO fact sheets")
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/raw/who_dataset.json",
        help="Output file path"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Delay between requests (seconds)"
    )
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=None,
        help="Maximum number of pages to scrape (for testing)"
    )
    
    args = parser.parse_args()
    
    scraper = WHOScraper(output_path=args.output)
    scraper.scrape_all(delay=args.delay, max_pages=args.max_pages)


if __name__ == "__main__":
    main()