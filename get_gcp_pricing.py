#!/usr/bin/env python3
"""
Complete GCP VM Pricing Data Fetcher and Analyzer

Fetches GCP pricing data via Cloud Billing API, processes Compute Engine VM pricing,
and generates comparison tables for sustained vs. spot pricing with discounts.

Usage:
    python3 get_gcp_vm_pricing.py              # Download and process data
    python3 get_gcp_vm_pricing.py --table      # Show formatted tables from existing data
    python3 get_gcp_vm_pricing.py --process    # Reprocess existing raw data

Requirements:
    - gcp-pricing.key file with GCP API key (for download mode)
    - pandas library
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from uuid import uuid4

import pandas as pd
import requests


def timing(comment):
    """Timing decorator that prints execution time with a custom comment."""

    def decorator(func):

        @wraps(func)
        def wrapper_func(*args, **kwargs):
            print(f"⏱️  Starting: {comment}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ Completed: {comment} (took {duration:.2f} seconds)")
            return result

        return wrapper_func

    return decorator


def load_api_key():
    """Load GCP API key from file."""
    try:
        with open('gcp-pricing.key', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("API key file 'gcp-pricing.key' not found")


@timing("Fetching paginated data from GCP API")
def fetch_paginated_data(url_base, api_key, data_key):
    """Fetch paginated data from GCP API."""
    print(f"Downloading {data_key}...")
    all_data = []
    page_token = ""
    part_num = 1

    while True:
        url = f"{url_base}?pageSize=5000&key={api_key}"
        if page_token:
            url += f"&pageToken={page_token}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise Exception(f"Error downloading {data_key}: {e}")

        all_data.extend(data.get(data_key, []))
        page_token = data.get('nextPageToken', '')
        print(
            f"Downloaded {data_key} part {part_num}{' (final part)' if not page_token else ''}"
        )
        if not page_token:
            break
        part_num += 1

    print(f"Total {data_key} entries downloaded: {len(all_data)}")
    return all_data


@timing("Combining pricing data with SKU metadata")
def combine_pricing_and_sku_data(pricing_data, sku_data):
    """Combine pricing data with SKU metadata."""
    print("Combining pricing data with SKU metadata...")

    sku_lookup = {sku['skuId']: sku for sku in sku_data}
    combined_data = []

    for i, price_entry in enumerate(pricing_data, 1):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(pricing_data)} entries...")

        price_name = price_entry.get('name', '')
        if '/price' not in price_name:
            continue

        sku_id = price_name.split('/')[1]
        if sku_id not in sku_lookup:
            continue

        sku = sku_lookup[sku_id]
        price_usd = price_entry.get('rate', {}).get('tiers', [{}])[0].get(
            'listPrice', {}).get('nanos', 0) / 1_000_000_000
        region = sku.get('geoTaxonomy',
                         {}).get('regionalMetadata',
                                 {}).get('region', {}).get('region', 'global')
        unit_info = price_entry.get('rate', {}).get('unitInfo', {})

        # Extract resource type from display name
        display_name = sku.get('displayName', '')
        resource_type = None

        # Check for VM resource types
        if 'Instance Core' in display_name or 'Predefined Instance Core' in display_name:
            resource_type = 'cpu'
        elif 'Instance Ram' in display_name or 'Predefined Instance Ram' in display_name:
            resource_type = 'ram'
        # Check for PD resource types
        elif extract_pd_info(display_name):
            resource_type = 'pd'

        entry = {
            'skuId': sku_id,
            'displayName': display_name,
            'region': region,
            'priceUSD': price_usd,
            'unit': unit_info.get('unit', ''),
            'unitDescription': unit_info.get('unitDescription', ''),
            'currencyCode': price_entry.get('currencyCode', 'USD')
        }

        # Add resource type field if identified
        if resource_type:
            entry['resourceType'] = resource_type

            # Add PD-specific metadata if it's a PD entry
            if resource_type == 'pd':
                pd_info = extract_pd_info(display_name)
                if pd_info:
                    entry.update(pd_info)

        combined_data.append(entry)

    print(f"Combined {len(combined_data)} entries with metadata")
    return combined_data


def extract_machine_info(display_name):
    """Extract machine family, resource type, and spot status from display name."""
    patterns = [
        r'(Spot Preemptible )?[A-Z0-9]+D? Instance (Core|Ram) running in',
        r'Vertex AI:.*?[A-Z0-9]+D? Predefined Instance (Core|Ram) running in'
    ]
    if not any(
            re.search(pattern, display_name, re.IGNORECASE)
            for pattern in patterns):
        return None

    skip_patterns = [
        'committed', 'colab', 'confidential', 'autopilot', 'workbench',
        'sole tenant', 'management fee', 'kubernetes', 'dataflow', 'dataproc',
        'neural', 'custom', 'premium', 'discount', 'tenancy', 'extended'
    ]
    if any(pattern in display_name.lower() for pattern in skip_patterns):
        return None

    machine_families = [
        'N1', 'N2', 'N2D', 'N4', 'E2', 'C2', 'C2D', 'C3', 'C3D', 'C4', 'C4A',
        'M1', 'M2', 'M3', 'T2D', 'T2A', 'A2', 'A3', 'G2', 'H3', 'Z3'
    ]
    machine_family = next(
        (family for family in machine_families
         if re.search(rf'\b{family}\b', display_name, re.IGNORECASE)), None)
    if not machine_family:
        return None

    resource_type = 'core' if 'Instance Core' in display_name or 'Predefined Instance Core' in display_name else 'ram'
    is_spot = bool(
        re.search(r'(spot|preemptible)', display_name, re.IGNORECASE))

    return {
        'machine_family': machine_family,
        'resource_type': resource_type,
        'is_spot': is_spot
    }


def extract_pd_info(display_name):
    """Extract PD type, scope (regional/zonal), and region from display name."""
    # Only process PD entries attached to VMs and snapshots
    pd_patterns = [
        r'Storage PD.*in\s+(.+)$',
        r'(Regional|Zonal)?\s*(Standard|Balanced|SSD|Extreme)?\s*PD\s*(Capacity|Instant Snapshot).*in\s+(.+)$',
        r'(Regional|Zonal)?\s*Storage PD.*in\s+(.+)$'
    ]

    if not any(re.search(pattern, display_name, re.IGNORECASE) for pattern in pd_patterns):
        return None

    # Skip non-VM PD entries
    skip_patterns = [
        'workbench', 'vertex', 'database', 'sql', 'spanner', 'datastore',
        'firestore', 'bigquery', 'bigtable', 'memorystore', 'redis',
        'dataproc', 'dataflow', 'composer', 'kubernetes', 'gke',
        'autopilot', 'management fee', 'committed'
    ]
    if any(pattern in display_name.lower() for pattern in skip_patterns):
        return None

    # Extract PD type (Standard, Balanced, SSD, Extreme)
    pd_type = 'standard'  # default
    if re.search(r'\bbalanced\b', display_name, re.IGNORECASE):
        pd_type = 'balanced'
    elif re.search(r'\bssd\b', display_name, re.IGNORECASE):
        pd_type = 'ssd'
    elif re.search(r'\bextreme\b', display_name, re.IGNORECASE):
        pd_type = 'extreme'

    # Extract scope (regional vs zonal)
    scope = 'zonal'  # default
    if re.search(r'\bregional\b', display_name, re.IGNORECASE):
        scope = 'regional'

    # Extract region from "in <region>" pattern
    region_match = re.search(r'in\s+(.+)$', display_name, re.IGNORECASE)
    region = region_match.group(1).strip() if region_match else 'unknown'

    # Determine if it's snapshot or capacity
    resource_subtype = 'capacity'
    if 'snapshot' in display_name.lower():
        resource_subtype = 'snapshot'

    return {
        'pd_type': pd_type,
        'scope': scope,
        'region_name': region,
        'resource_subtype': resource_subtype
    }


@timing("Processing VM pricing data for analysis")
def process_vm_pricing_data(data):
    """Process combined data to extract VM pricing comparisons."""
    print(f"Processing {len(data)} entries for VM pricing analysis...")

    parsed_data = []
    for item in data:
        machine_info = extract_machine_info(item['displayName'])
        if machine_info:
            parsed_data.append({**item, **machine_info})
    print(f"Found {len(parsed_data)} relevant VM pricing entries")

    grouped = defaultdict(lambda: {
        'sustained': None,
        'spot': None,
        'sustained_sku': None,
        'spot_sku': None
    })

    for item in parsed_data:
        key = (item['machine_family'], item['region'], item['resource_type'])
        price_type = 'spot' if item['is_spot'] else 'sustained'
        grouped[key][price_type] = item['priceUSD']
        grouped[key][f'{price_type}_sku'] = item['skuId']

    results = [{
        'Machine Family':
        machine_family,
        'Region':
        region,
        'Resource':
        resource_type,
        'Sustained SKU':
        prices['sustained_sku'],
        'Sustained Price':
        f"${prices['sustained']:.6f}",
        'Spot SKU':
        prices['spot_sku'],
        'Spot Price':
        f"${prices['spot']:.6f}",
        'Discount %':
        f"{((prices['sustained'] - prices['spot']) / prices['sustained'] * 100):.1f}%"
    } for (machine_family, region, resource_type), prices in grouped.items()
               if prices['sustained'] and prices['spot']
               and prices['sustained'] > 0]

    return results


@timing("Processing PD pricing data for analysis")
def process_pd_pricing_data(data):
    """Process combined data to extract PD pricing information."""
    print(f"Processing {len(data)} entries for PD pricing analysis...")

    parsed_data = []
    for item in data:
        if item.get('resourceType') == 'pd':
            parsed_data.append(item)

    print(f"Found {len(parsed_data)} PD pricing entries")

    # Group by PD type, scope, region, and resource subtype
    grouped = defaultdict(list)
    for item in parsed_data:
        key = (
            item.get('pd_type', 'unknown'),
            item.get('scope', 'unknown'),
            item['region'],
            item.get('resource_subtype', 'unknown')
        )
        grouped[key].append(item)

    results = []
    for (pd_type, scope, region, subtype), items in grouped.items():
        if items:
            # Take the first item as representative (they should have same price)
            item = items[0]
            results.append({
                'PD Type': pd_type.upper() if pd_type == 'ssd' else pd_type.title(),
                'Scope': scope.title(),
                'Region': region,
                'Resource': subtype.title(),
                'SKU ID': item['skuId'],
                'Price USD': f"${item['priceUSD']:.6f}",
                'Unit': item['unit'],
                'Unit Description': item['unitDescription']
            })

    return results


def save_processed_data(timestamp, processed_results):
    """Save processed results to CSV and return DataFrame."""
    df = pd.DataFrame(processed_results).sort_values(
        ['Machine Family', 'Region', 'Resource'])
    csv_file = f"{timestamp}-vm-pricing-comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"VM pricing comparison saved to: {csv_file}")
    return df


def print_summary(df):
    """Print summary statistics and sample data."""
    print("\nVM Pricing Comparison Summary:")
    print("=" * 60)
    print(f"- Total comparisons: {len(df)}")
    print(f"- Machine families: {df['Machine Family'].nunique()}")
    print(f"- Regions: {df['Region'].nunique()}")
    print(
        f"- Average spot discount: {df['Discount %'].str.rstrip('%').astype(float).mean():.1f}%"
    )
    print(
        f"- Max spot discount: {df['Discount %'].str.rstrip('%').astype(float).max():.1f}%"
    )
    print(
        f"- Min spot discount: {df['Discount %'].str.rstrip('%').astype(float).min():.1f}%"
    )

    print("\nMachine families included:")
    for family, count in df['Machine Family'].value_counts().items():
        print(f"- {family}: {count} region/resource combinations")

    print("\nSample pricing data:")
    print("-" * 60)
    print(
        df.head(5)[[
            'Machine Family', 'Region', 'Resource', 'Sustained Price',
            'Spot Price', 'Discount %'
        ]].to_string(index=False))


def display_vm_tables(csv_file):
    """Display formatted VM tables by machine family from CSV."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find VM pricing data file: {csv_file}")
        print("Run the script without --vm first to download data.")
        return 1

    print("GCP VM Pricing Comparison Tables")
    print("=" * 80)
    print(f"Data from: {csv_file}")
    print(f"Total entries: {len(df)}\n")

    for family in sorted(df['Machine Family'].unique()):
        family_df = df[df['Machine Family'] == family]
        print(f"{family} Machine Family ({len(family_df)} entries):")
        print("-" * 80)
        print(family_df[[
            'Region', 'Resource', 'Sustained SKU', 'Sustained Price',
            'Spot SKU', 'Spot Price', 'Discount %'
        ]].to_string(index=False))
        print()

    return 0


def display_pd_tables(csv_file):
    """Display formatted PD tables from CSV."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find PD pricing data file: {csv_file}")
        print("Run the script without --pd first to download data.")
        return 1

    print("GCP Persistent Disk Pricing Tables")
    print("=" * 80)
    print(f"Data from: {csv_file}")
    print(f"Total entries: {len(df)}\n")

    for pd_type in sorted(df['PD Type'].unique()):
        type_df = df[df['PD Type'] == pd_type]
        print(f"{pd_type} Persistent Disk ({len(type_df)} entries):")
        print("-" * 80)
        print(type_df[[
            'Scope', 'Region', 'Resource', 'Price USD', 'Unit Description'
        ]].to_string(index=False))
        print()

    return 0


def find_latest_file(suffix):
    """Find the most recent file with the given suffix."""
    files = [
        f for f in os.listdir('.') if f.endswith(suffix) and f.startswith('2')
    ]
    return sorted(files)[-1] if files else None


def load_raw_data(filename):
    """Load raw pricing data from JSON file."""
    print(f"Loading existing raw data from {filename}...")
    with open(filename, 'r') as f:
        return json.load(f)


@timing("Download - fetching and processing fresh data")
def run_download():
    """Download and process fresh data."""
    print("GCP VM Pricing Data Fetcher and Analyzer")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d")
    api_key = load_api_key()

    # Fetch pricing and SKU data in parallel using threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        pricing_future = executor.submit(
            fetch_paginated_data,
            "https://cloudbilling.googleapis.com/v1beta/skus/-/prices",
            api_key, "prices")
        sku_future = executor.submit(
            fetch_paginated_data,
            "https://cloudbilling.googleapis.com/v2beta/skus", api_key, "skus")

        pricing_data = pricing_future.result()
        sku_data = sku_future.result()

    # Save raw data in separate timestamped files
    pricing_filename = f"{timestamp}-raw-pricing-data.json"
    with open(pricing_filename, 'w') as f:
        json.dump(pricing_data, f, indent=2)
    print(
        f"Raw pricing data saved to: {pricing_filename} ({len(pricing_data)} entries)"
    )

    sku_filename = f"{timestamp}-raw-sku-data.json"
    with open(sku_filename, 'w') as f:
        json.dump(sku_data, f, indent=2)
    print(f"Raw SKU data saved to: {sku_filename} ({len(sku_data)} entries)")

    combined_data = combine_pricing_and_sku_data(pricing_data, sku_data)

    raw_filename = f"{timestamp}-compute-pricing.json"
    with open(raw_filename, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Combined pricing data saved to: {raw_filename}")

    # Process VM data
    vm_df = save_processed_data(timestamp, process_vm_pricing_data(combined_data))
    print_summary(vm_df)

    # Process PD data
    pd_results = process_pd_pricing_data(combined_data)
    if pd_results:
        pd_df = pd.DataFrame(pd_results).sort_values(['PD Type', 'Scope', 'Region', 'Resource'])
        pd_csv_file = f"{timestamp}-pd-pricing-comparison.csv"
        pd_df.to_csv(pd_csv_file, index=False)
        print(f"\n📊 PD pricing data saved to: {pd_csv_file} ({len(pd_results)} entries)")

    print(f"\n✅ Complete! Pricing data for {timestamp} is ready.")
    print(f"📊 VM data: {timestamp}-vm-pricing-comparison.csv")
    if pd_results:
        print(f"📊 PD data: {timestamp}-pd-pricing-comparison.csv")
    return 0


@timing("Process - reprocessing existing raw data")
def run_process():
    """Reprocess existing raw data."""
    # Look for raw pricing and SKU data files
    pricing_file = find_latest_file('-raw-pricing-data.json')
    sku_file = find_latest_file('-raw-sku-data.json')

    if pricing_file and sku_file:
        print("GCP Pricing Data Reprocessor")
        print("=" * 50)
        print(f"Using raw pricing data: {pricing_file}")
        print(f"Using raw SKU data: {sku_file}")

        timestamp = datetime.now().strftime("%Y%m%d")

        # Load raw data files
        pricing_data = load_raw_data(pricing_file)
        sku_data = load_raw_data(sku_file)

        # Recombine with updated logic (including resource type and PD support)
        combined_data = combine_pricing_and_sku_data(pricing_data, sku_data)

        # Save updated compute pricing file
        raw_filename = f"{timestamp}-compute-pricing.json"
        with open(raw_filename, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Updated combined pricing data saved to: {raw_filename}")

        # Process VM data
        vm_df = save_processed_data(timestamp, process_vm_pricing_data(combined_data))
        print_summary(vm_df)

        # Process PD data
        pd_results = process_pd_pricing_data(combined_data)
        if pd_results:
            pd_df = pd.DataFrame(pd_results).sort_values(['PD Type', 'Scope', 'Region', 'Resource'])
            pd_csv_file = f"{timestamp}-pd-pricing-comparison.csv"
            pd_df.to_csv(pd_csv_file, index=False)
            print(f"\n📊 PD pricing data saved to: {pd_csv_file} ({len(pd_results)} entries)")

        print(f"\n✅ Complete! Reprocessed pricing data for {timestamp}")
        print(f"📊 VM data: {timestamp}-vm-pricing-comparison.csv")
        if pd_results:
            print(f"📊 PD data: {timestamp}-pd-pricing-comparison.csv")
        return 0
    else:
        # Fall back to existing behavior if raw files not found
        raw_data_file = find_latest_file('-compute-pricing.json')
        if not raw_data_file:
            print("❌ Error: No raw pricing data files found.")
            print("Run the script without --process first to download data.")
            return 1

        print("GCP Pricing Data Reprocessor")
        print("=" * 50)
        print(f"Using existing combined data: {raw_data_file}")

        timestamp = datetime.now().strftime("%Y%m%d")
        combined_data = load_raw_data(raw_data_file)

        # Process VM data
        vm_df = save_processed_data(timestamp, process_vm_pricing_data(combined_data))
        print_summary(vm_df)

        # Process PD data
        pd_results = process_pd_pricing_data(combined_data)
        if pd_results:
            pd_df = pd.DataFrame(pd_results).sort_values(['PD Type', 'Scope', 'Region', 'Resource'])
            pd_csv_file = f"{timestamp}-pd-pricing-comparison.csv"
            pd_df.to_csv(pd_csv_file, index=False)
            print(f"\n📊 PD pricing data saved to: {pd_csv_file} ({len(pd_results)} entries)")

        print(f"\n✅ Complete! Reprocessed pricing data for {timestamp}")
        print(f"📊 VM data: {timestamp}-vm-pricing-comparison.csv")
        if pd_results:
            print(f"📊 PD data: {timestamp}-pd-pricing-comparison.csv")
        return 0


@timing("Main execution")
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='GCP VM Pricing Data Fetcher and Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 get_gcp_pricing.py              # Download and process data
  python3 get_gcp_pricing.py --vm         # Show formatted VM tables from existing data
  python3 get_gcp_pricing.py --pd         # Show formatted PD tables from existing data
  python3 get_gcp_pricing.py --process    # Reprocess existing raw data
        """)
    parser.add_argument('--vm',
                        action='store_true',
                        help='Display formatted VM tables from existing CSV data')
    parser.add_argument('--pd',
                        action='store_true', 
                        help='Display formatted PD tables from existing CSV data')
    parser.add_argument('--process',
                        action='store_true',
                        help='Reprocess existing raw data without downloading')

    args = parser.parse_args()

    try:
        if args.vm:
            csv_file = find_latest_file('-vm-pricing-comparison.csv')
            return display_vm_tables(csv_file) if csv_file else 1
        elif args.pd:
            csv_file = find_latest_file('-pd-pricing-comparison.csv')
            return display_pd_tables(csv_file) if csv_file else 1
        return run_process() if args.process else run_download()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
