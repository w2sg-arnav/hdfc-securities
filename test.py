import requests
import os
import zipfile
import shutil
from pathlib import Path
import time
import logging
import json
import random
from typing import Optional, Dict, Any
from datetime import datetime

class RHPDownloader:
    def __init__(self, api_key: str, debug: bool = False):
        self.api_key = api_key
        self.api_url = "https://api.ipoalerts.in/ipos"
        self.api_headers = {"x-api-key": api_key}

        # Setup directories
        self.base_dir = Path("RHP_Documents")
        self.base_dir.mkdir(exist_ok=True)
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.debug = debug

        # Setup logging
        self._setup_logging()

        # Setup browser session
        self.session = self._setup_session()

    def _setup_logging(self):
        """Configure logging with file and console output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"rhp_downloader_{timestamp}.log"

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

    def is_valid_rhp(self, file_path: Path) -> bool:
        """Check if file is likely to be an RHP based on size."""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            is_valid = 3.5 <= size_mb <= 1000
            if self.debug:
                logging.debug(f"File size: {size_mb:.2f} MB - Valid RHP: {is_valid}")
            return is_valid
        except Exception as e:
            logging.error(f"Error checking file size: {str(e)}")
            return False

    def process_ipo(self, ipo_data: Optional[Dict[str, Any]]) -> None:
        """Process a single IPO entry."""
        if not ipo_data:
            logging.warning("Received empty IPO data.")
            return

        company_name = ipo_data.get("name", "Unknown_Company").replace("/", "-").strip()
        prospectus_url = ipo_data.get("prospectusUrl")

        if not prospectus_url:
            logging.warning(f"No prospectus URL for {company_name}")
            return

        if self.debug:
            logging.debug(f"Processing {company_name}: {prospectus_url}")

        temp_file = self.temp_dir / f"temp_{company_name}"
        if not self.download_file(prospectus_url, temp_file):
            return

        try:
            if prospectus_url.lower().endswith('.pdf'):
                if self.is_valid_rhp(temp_file):
                    final_path = self.base_dir / f"{company_name}_RHP.pdf"
                    shutil.move(temp_file, final_path)
                    size_mb = final_path.stat().st_size / (1024 * 1024)
                    logging.info(f"Saved PDF as RHP for {company_name} ({size_mb:.2f} MB)")
                else:
                    logging.warning(f"Downloaded PDF for {company_name} is not a valid RHP")

            elif prospectus_url.lower().endswith('.zip'):
                self.extract_rhp_from_zip(temp_file, company_name)

            else:
                logging.warning(f"Unsupported file type for {company_name}: {prospectus_url}")

        finally:
            temp_file.unlink(missing_ok=True)

    def extract_rhp_from_zip(self, zip_path: Path, company_name: str) -> Optional[Path]:
        """Extract RHP from ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
                if not pdf_files:
                    logging.warning(f"No PDF files found in ZIP for {company_name}")
                    return None

                for pdf in pdf_files:
                    temp_path = self.temp_dir / "temp.pdf"
                    with zip_ref.open(pdf) as source, open(temp_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                    if self.is_valid_rhp(temp_path):
                        final_path = self.base_dir / f"{company_name}_RHP.pdf"
                        shutil.move(temp_path, final_path)
                        logging.info(f"Extracted RHP for {company_name}")
                        return final_path

            logging.warning(f"No valid RHP found in ZIP for {company_name}")
            return None

        except Exception as e:
            logging.error(f"Error processing ZIP file for {company_name}: {str(e)}")
            return None

    def download_rhps(self, status: str) -> None:
        """Download RHPs for all IPOs with given status."""
        try:
            response = requests.get(self.api_url, headers=self.api_headers, params={"status": status}, timeout=30)
            response.raise_for_status()

            ipo_data = response.json().get("ipos", [])
            if not ipo_data:
                logging.info(f"No IPOs found with status: {status}")
                return

            logging.info(f"Processing {len(ipo_data)} IPOs with status: {status}")
            for ipo in ipo_data:
                self.process_ipo(ipo)

        except Exception as e:
            logging.error(f"Error fetching IPO data: {str(e)}")

        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir.mkdir(exist_ok=True)


def main():
    api_key = os.getenv("IPO_API_KEY") or input("Enter your API key: ").strip()
    debug_mode = os.getenv("DEBUG", "").lower() == "true"

    downloader = RHPDownloader(api_key, debug=debug_mode)
    valid_statuses = ["upcoming", "closed", "open", "announced", "listed"]

    while True:
        status = input(f"Enter IPO status ({'/'.join(valid_statuses)}): ").strip().lower()
        if status in valid_statuses:
            break
        print(f"Invalid status. Please choose from: {', '.join(valid_statuses)}")

    downloader.download_rhps(status)


if __name__ == "__main__":
    main()
