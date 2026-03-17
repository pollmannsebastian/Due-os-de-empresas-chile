"""
Web Scraper for Diario Oficial
===============================
Run: python etl/scraper_efficient.py
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import datetime
import os
import io
import logging
import concurrent.futures
import threading
import re
from pypdf import PdfReader

try:
    import ujson as fastjson
except ImportError:
    import json as fastjson

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://www.diariooficial.interior.gob.cl/edicionelectronica"
COMPLETED_DATES_FILE = os.path.join(BASE_DIR, "completed_dates.txt")
DATASET_FILE = os.path.join(BASE_DIR, "dataset_optimized_v2.jsonl")

# Thread lock for safely appending to the dataset file
write_lock = threading.Lock()
existing_cves = set()


def get_session():
    """
    Creates a session with retries and connection pooling.
    """
    session = requests.Session()
    retry = Retry(connect=3, read=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_edition(date, session=None):
    if session is None: session = requests
    date_str = date.strftime("%d-%m-%Y")
    url = f"{BASE_URL}/select_edition.php?date={date_str}"
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        editions = []
        for link in links:
            href = link['href']
            if 'index.php' in href and 'edition=' in href:
                try:
                    params = dict(x.split('=') for x in href.split('?')[1].split('&'))
                    if params.get('date') == date_str:
                        editions.append({
                            'edition': params.get('edition'),
                            'v': params.get('v', '1'),
                            'date': date_str
                        })
                except:
                    pass
        return editions
    except Exception as e:
        logging.error(f"Error getting edition for {date_str}: {e}")
        return []

def get_company_pdf_links(edition_info, session=None):
    if session is None: session = requests
    date_str = edition_info['date']
    edition = edition_info['edition']
    v = edition_info['v']
    
    url = f"{BASE_URL}/empresas_cooperativas.php?date={date_str}&edition={edition}&v={v}"
    
    pdf_links = []
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200: return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if href.lower().endswith('.pdf') and 'publicaciones' in href:
                text = link.get_text(strip=True)
                full_url = href if href.startswith('http') else f"https://www.diariooficial.interior.gob.cl{href}"
                
                pdf_links.append({
                    'url': full_url,
                    'title': text,
                    'edition_info': edition_info
                })
        
        return pdf_links
    except Exception as e:
        logging.error(f"Error getting PDF links for {date_str} ed {edition}: {e}")
        return []

def process_pdf(pdf_info, session=None):
    if session is None: session = requests
    url = pdf_info['url']
    cve = url.split('/')[-1].replace('.pdf', '')
    date_str = pdf_info['edition_info']['date']
    
    # Check if already processed
    if cve in existing_cves:
        logging.info(f"Skipping {cve}, already in dataset dataset_optimized_v2.jsonl")
        return True

    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200: 
             logging.error(f"Failed to download {url}: Status {response.status_code}")
             return False
        
        pdf_file = io.BytesIO(response.content)
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as pdf_err:
             logging.error(f"Corrupt PDF {cve}: {pdf_err}")
             return False
            
        data = {
            'cve': cve,
            'title': pdf_info['title'],
            'url': url,
            'date': date_str,
            'edition': pdf_info['edition_info']['edition'],
            'content': text.strip()
        }
        
        if fastjson.__name__ == 'ujson':
            json_line = fastjson.dumps(data, ensure_ascii=True, escape_forward_slashes=False)
        else:
            json_line = fastjson.dumps(data)
            
        with write_lock:
            with open(DATASET_FILE, 'a', encoding='utf-8') as f:
                f.write(json_line + '\n')
            existing_cves.add(cve)
            
        logging.info(f"Appended {cve} ({date_str}) to dataset.")
        return True
        
    except Exception as e:
        logging.error(f"Error processing PDF {url}: {e}")
        return False

def process_date(date, completed_dates):
    date_str = date.strftime("%d-%m-%Y")
    
    if date_str in completed_dates:
        logging.info(f"Skipping date {date_str} (Marked as Completed)")
        return True

    s = get_session()
    
    try:
        editions = get_edition(date, session=s)
        if not editions:
            logging.info(f"No edition found for {date_str}")
            return True
            
        all_pdfs = []
        for edition in editions:
            pdfs = get_company_pdf_links(edition, session=s)
            logging.info(f"Found {len(pdfs)} PDFs for {date_str} Edition {edition['edition']}")
            all_pdfs.extend(pdfs)
        
        if not all_pdfs:
            return True

        success_count = 0
        for pdf in all_pdfs:
            if process_pdf(pdf, session=s):
                success_count += 1
                
        is_complete = (success_count == len(all_pdfs))
        
        if is_complete and date < datetime.date.today():
            with write_lock:
                with open(COMPLETED_DATES_FILE, "a") as f:
                    f.write(date_str + "\n")
                
        return is_complete
    finally:
        s.close()
 
def main():
    start_date = datetime.date(2017, 1, 1)
    today = datetime.date.today()
        
    completed_dates = set()
    if os.path.exists(COMPLETED_DATES_FILE):
        with open(COMPLETED_DATES_FILE, "r") as f:
            completed_dates = set(line.strip() for line in f if line.strip())
            
    # Load existing CVEs from dataset to avoid re-downloading
    if os.path.exists(DATASET_FILE):
        logging.info("Loading existing CVEs from dataset file... This might take a moment.")
        cve_pattern = re.compile(r'"cve"\s*:\s*"([^"]+)"')
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                match = cve_pattern.search(line)
                if match:
                    existing_cves.add(match.group(1))
        logging.info(f"Loaded {len(existing_cves)} existing CVEs.")
            
    dates = []
    current = start_date
    while current <= today:
        dates.append(current)
        current += datetime.timedelta(days=1)
    
    logging.info(f"Starting crawl for {len(dates)} dates from {start_date} to {today}")
    logging.info(f"Already completed {len(completed_dates)} dates.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(lambda d: process_date(d, completed_dates), dates))
        
    logging.info("Done.")

if __name__ == "__main__":
    main()
