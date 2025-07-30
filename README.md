# GCP Pricing Extractor

A quick Python script for extracting GCP VM and persistent disk pricing data from the Google Cloud Billing API.

**Note: This is just a simple proof-of-concept script and not particularly useful in its current form.**

## Setup

1. Get a GCP API key with access to the Cloud Billing API
2. Save the API key to `gcp-pricing.key` in the project root (this file is gitignored)
3. Run the script: `./get_gcp_vm_pricing.py`

## Usage

- `./get_gcp_vm_pricing.py` - Download fresh pricing data from GCP API
- `./get_gcp_vm_pricing.py --process` - Reprocess existing cached data files

## Output Files

- `YYYYMMDD-raw-pricing-data.json` - Raw pricing data from GCP API
- `YYYYMMDD-raw-sku-data.json` - Raw SKU metadata from GCP API
- `YYYYMMDD-compute-pricing.json` - Processed VM pricing with resource types (cpu/ram)
- `YYYYMMDD-pd-pricing.json` - Processed persistent disk pricing

## API Key Required

You must obtain a Google Cloud API key with Cloud Billing API access and place it in `gcp-pricing.key`. The script will not work without this key file.

## Examples

### VM Table

```
$ ./get_gcp_pricing.py --vm | grep -A 30 N2D
[...]
E2 Machine Family (82 entries):
--------------------------------------------------------------------------------
                 Region Resource  Sustained SKU Sustained Price       Spot SKU Spot Price Discount %
          africa-south1     core DC64-21B2-0622       $0.028530 67C8-70B6-CA08  $0.007260      74.6%
          africa-south1      ram 7B3A-CCB8-2588       $0.004400 635A-FEFA-74F8  $0.000973      77.9%
             asia-east1     core 92C8-7C92-6AEF       $0.025255 6973-FC9A-3477  $0.010100      60.0%
             asia-east1      ram FEE3-AF56-8EDF       $0.004062 64F3-9E6F-1C3E  $0.001354      66.7%
             asia-east2     core 8EEE-9FBE-A2D2       $0.036623 6210-27D9-E735  $0.004320      88.2%

[...]
N2D Machine Family (78 entries):
--------------------------------------------------------------------------------
                 Region Resource  Sustained SKU Sustained Price       Spot SKU Spot Price Discount %
          africa-south1     core A903-CE02-AD33       $0.035973 71BB-8DAF-800B  $0.010730      70.2%
          africa-south1      ram D455-945C-A274       $0.004821 F56A-7453-C8B4  $0.001441      70.1%
             asia-east1     core EF6C-66BA-C9C6       $0.031844 6B4A-3961-E53F  $0.012740      60.0%
             asia-east1      ram FFF7-7A42-F2E8       $0.004268 672C-2EB5-278B  $0.001707      60.0%
             asia-east2     core A640-62BC-0C6C       $0.038481 7B34-6F0F-A857  $0.009390      75.6%
             asia-east2      ram E1D7-37E7-B521       $0.005931 14B6-07DE-0BDC  $0.001253      78.9%
[...]
```

### PD Table

```
[...]
SSD Persistent Disk (170 entries):
--------------------------------------------------------------------------------
   Scope                  Region Resource Price USD Unit Description
Regional           africa-south1 Capacity $0.444720   gibibyte month
Regional           africa-south1 Snapshot $0.444700   gibibyte month
[...]

Standard Persistent Disk (171 entries):
--------------------------------------------------------------------------------
   Scope                  Region Resource Price USD Unit Description
Regional           africa-south1 Capacity $0.104640   gibibyte month
Regional           africa-south1 Snapshot $0.104600   gibibyte month
Regional              asia-east1 Snapshot $0.080000   gibibyte month
[...]
   Zonal           africa-south1 Capacity $0.000000   gibibyte month
   Zonal           africa-south1 Snapshot $0.052300   gibibyte month
   Zonal              asia-east1 Snapshot $0.040000   gibibyte month
[...]
```

### JSON

```json
[...]
 {
    "skuId": "0428-415B-93BE",
    "displayName": "Spot Preemptible E2 Instance Ram running in Osaka",
    "region": "asia-northeast2",
    "priceUSD": 0.001163,
    "unit": "GiBy.h",
    "unitDescription": "gibibyte hour",
    "currencyCode": "USD",
    "resourceType": "ram"
  },
  {
    "skuId": "0429-3D5C-ED01",
    "displayName": "Storage PD Snapshot in Dammam",
    "region": "me-central2",
    "priceUSD": 0.08,
    "unit": "GiBy.mo",
    "unitDescription": "gibibyte month",
    "currencyCode": "USD",
    "resourceType": "pd",
    "pd_type": "standard",
    "scope": "zonal",
    "region_name": "Dammam",
    "resource_subtype": "snapshot"
  },
[...]
```
