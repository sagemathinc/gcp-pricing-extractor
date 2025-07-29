## Anaconda Commands

- To call/activate anaconda: `source /home/hsy/s/anaconda/etc/profile.d/conda.sh && conda activate base`
- To create a new conda environment: `conda create --name <environment_name>`
- To list existing conda environments: `conda env list`

## GCP Pricing Tool

### Main Script
- `get_gcp_vm_pricing.py` - Main script for fetching and processing GCP pricing data
- Run with `./get_gcp_vm_pricing.py` to download fresh data
- Use `--process` flag to reprocess existing cached raw data files

### Data Files
- `YYYYMMDD-raw-pricing-data.json` - Raw pricing data from GCP API
- `YYYYMMDD-raw-sku-data.json` - Raw SKU metadata from GCP API  
- `YYYYMMDD-compute-pricing.json` - Processed pricing data with resource types
- `YYYYMMDD-pd-pricing.json` - Processed persistent disk pricing data

### Key Features
- Extracts VM instance pricing (CPU cores and RAM) with resource type identification
- Extracts persistent disk pricing including Hyperdisk varieties
- Adds `resourceType` field to distinguish between "cpu" and "ram" entries
- Pattern matching for different disk types: standard, balanced, SSD, hyperdisk-balanced, hyperdisk-extreme, hyperdisk-throughput