"""Script to download and assemble the raw TSA passenger throughput dataset into data/raw/tsa_raw.csv."""
import urllib.request
import json
import concurrent.futures
import re
import socket
from pathlib import Path
from collections import defaultdict
import pandas as pd

socket.setdefaulttimeout(8)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "tsa_raw.csv"

def fetch_json_list():
    url = 'https://api.github.com/repos/mikelor/TsaThroughput/git/trees/main?recursive=1'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    tree = json.loads(res.read().decode('utf-8')).get('tree', [])
    return [item['path'] for item in tree if item['path'].startswith('data/raw/tsa/throughput/') and item['path'].endswith('.json')]

def clean_date_str(dt_str: str) -> str:
    dt_str = dt_str[:10]
    if dt_str.startswith('0201-'):
        return '2021-' + dt_str[5:]
    if dt_str.startswith('0202-'):
        return '2022-' + dt_str[5:]
    if dt_str.startswith('0200-'):
        return '2020-' + dt_str[5:]
    return dt_str

def process_file(path):
    raw_url = f'https://raw.githubusercontent.com/mikelor/TsaThroughput/main/{path}'
    try:
        r = urllib.request.Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
        content = urllib.request.urlopen(r, timeout=8).read().decode('utf-8')
        data = json.loads(content)
        local_totals = defaultdict(int)
        for airport in data.get('Airports', []):
            for day in airport.get('Days', []):
                dt_str = clean_date_str(day.get('Date', ''))
                if re.match(r'^\d{4}-\d{2}-\d{2}$', dt_str):
                    for cp in day.get('Checkpoints', []):
                        for hr in cp.get('Hours', []):
                            local_totals[dt_str] += hr.get('Amount', 0)
        return local_totals
    except Exception as e:
        return {}

def main():
    print("Fetching file index...", flush=True)
    json_paths = fetch_json_list()
    print(f"Processing {len(json_paths)} JSON throughput files...", flush=True)
    
    daily_totals = defaultdict(int)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_file, p): p for p in json_paths}
        for future in concurrent.futures.as_completed(futures):
            res_dict = future.result()
            done += 1
            if done % 50 == 0 or done == len(json_paths):
                print(f"Progress: {done}/{len(json_paths)} files processed", flush=True)
            for d, count in res_dict.items():
                daily_totals[d] += count

    df = pd.DataFrame(list(daily_totals.items()), columns=['Date', 'Passenger_Count'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Filter valid range: from 2019-01-01 onwards
    df = df[df['Date'] >= '2019-01-01'].sort_values('Date').reset_index(drop=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Successfully saved {len(df)} daily records to {RAW_PATH}", flush=True)
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}", flush=True)

if __name__ == '__main__':
    main()
