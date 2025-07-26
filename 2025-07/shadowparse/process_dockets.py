
import csv
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

# Create a directory for caching if it doesn't exist
CACHE_DIR = 'cached_dockets'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_docket_info(docket_number):
    cache_path = os.path.join(CACHE_DIR, f"{docket_number}.html")
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    url = f"https://www.supremecourt.gov/docket/docketfiles/html/public/{docket_number}.html"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        html_content = response.text
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return html_content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_for_outcome(html_content):
    if not html_content:
        return "Error fetching page"
    soup = BeautifulSoup(html_content, 'html.parser')
    proceedings_table = soup.find('table', {'id': 'proceedings'})
    #print(proceedings_table)
    if proceedings_table:
        text = proceedings_table.get_text().lower()
        print(text)
        if "denied by" in text:
            return "Denied"
        elif "granted" in text:
            return "Granted"
    return "Outcome not found"

def main():
    with open('items.csv', 'r') as infile, open('docket_outcomes.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile)
        next(reader)  # Skip header row
        writer = csv.writer(outfile)
        writer.writerow(['Docket Number', 'Outcome'])

        for row in reader:
            docket_number = row[0]
            #print(f"Processing {docket_number}...")
            html_content = get_docket_info(docket_number)
            outcome = parse_for_outcome(html_content)
            writer.writerow([docket_number, outcome])
            #print(f"  -> {outcome}")

if __name__ == "__main__":
    main()
