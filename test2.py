import os
import requests
import zipfile
import shutil
import logging
from pathlib import Path
import json
from datetime import datetime
import random
import time
from typing import Optional
from bs4 import BeautifulSoup

class NSEDataExtractor:
    def __init__(self, debug: bool = False):
        self.base_url = "https://www.nseindia.com/products/content/equities/ipos/ratio_basis_issues.htm"
        self.base_dir = Path("NSE_RHP_Documents")
        self.base_dir.mkdir(exist_ok=True)
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Set debug mode and logging
        self.debug = debug
        self._setup_logging()

        # Session setup
        self.session = self._setup_session()

        # Load previously processed IPOs (from a JSON file)
        self.processed_ipos_file = self.base_dir / "processed_ipos.json"
        self.processed_ipos = self._load_processed_ipos()

    def _setup_logging(self):
        """Configure logging with file and console output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"nse_data_extractor_{timestamp}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _setup_session(self) -> requests.Session:
        """Setup requests session with browser-like headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session

    def _random_sleep(self):
        """Introduce random sleep intervals between download attempts."""
        sleep_time = random.uniform(5, 15)
        logging.debug(f"Sleeping for {sleep_time:.2f} seconds to avoid rate limiting.")
        time.sleep(sleep_time)

    def _load_processed_ipos(self) -> dict:
        """Load the list of previously processed IPOs from a JSON file."""
        if self.processed_ipos_file.exists():
            with open(self.processed_ipos_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_ipos(self):
        """Save the list of processed IPOs to a JSON file."""
        with open(self.processed_ipos_file, 'w') as f:
            json.dump(self.processed_ipos, f)

    def get_zip_url_from_page(self, company_code: str) -> Optional[str]:
        """Scrape the NSE page for the ZIP file URL."""
        try:
            url = f"{self.base_url}?symbol={company_code}&series=SME&type=Active"
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            zip_link_tag = soup.find('a', href=True, text='Download Document')  # Adjust text as needed
            if zip_link_tag:
                return zip_link_tag['href']
            else:
                logging.warning(f"ZIP file link not found for company {company_code}")
                return None
        except Exception as e:
            logging.error(f"Error scraping the page for {company_code}: {str(e)}")
            return None

    def download_file(self, url: str, save_path: Path) -> bool:
        """Download file with retry mechanism and random sleep intervals."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Successfully downloaded file: {save_path}")
                return True

            except Exception as e:
                logging.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    self._random_sleep()
                else:
                    logging.error(f"Failed to download after {max_retries} attempts.")
                    return False
        return False

    def extract_rhp_from_zip(self, zip_path: Path, company_code: str) -> Optional[Path]:
        """Extract RHP from ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
                if not pdf_files:
                    logging.warning(f"No PDF files found in ZIP for {company_code}")
                    return None

                for pdf in pdf_files:
                    temp_path = self.temp_dir / "temp.pdf"
                    with zip_ref.open(pdf) as source, open(temp_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                    final_path = self.base_dir / f"{company_code}_RHP.pdf"
                    shutil.move(temp_path, final_path)
                    logging.info(f"Extracted RHP for {company_code}")
                    return final_path

            logging.warning(f"No valid RHP found in ZIP for {company_code}")
            return None

        except Exception as e:
            logging.error(f"Error processing ZIP file for {company_code}: {str(e)}")
            return None

    def download_rhps_for_new_companies(self, company_codes: list) -> None:
        """Download RHPs for only new companies that have not been processed."""
        for company_code in company_codes:
            if company_code not in self.processed_ipos:
                logging.info(f"Processing new company: {company_code}...")
                zip_url = self.get_zip_url_from_page(company_code)

                if zip_url:
                    # Download the ZIP file
                    temp_zip_path = self.temp_dir / f"{company_code}.zip"
                    if self.download_file(zip_url, temp_zip_path):
                        self.extract_rhp_from_zip(temp_zip_path, company_code)
                        self.processed_ipos[company_code] = True  # Mark as processed
                        self._save_processed_ipos()
                        temp_zip_path.unlink(missing_ok=True)  # Clean up ZIP file

                self._random_sleep()
            else:
                logging.info(f"Company {company_code} has already been processed.")

def main():
    debug_mode = input("Enable debug mode? (y/n): ").strip().lower() == 'y'
    downloader = NSEDataExtractor(debug=debug_mode)

    company_codes = input("Enter the company codes (comma-separated): ").strip().split(',')

    downloader.download_rhps_for_new_companies([code.strip() for code in company_codes])

if __name__ == "__main__":
    main()
