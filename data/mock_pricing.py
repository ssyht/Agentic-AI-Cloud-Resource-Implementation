"""
Mock cloud pricing data for AWS, GCP, and Azure.
Replace the PRICING_CATALOG entries with live API calls when credentials are available.

Live API references:
  AWS:   boto3 -> pricing.get_products()
  GCP:   google-cloud-billing -> CloudCatalogClient
  Azure: azure-mgmt-commerce -> RateCardClient
"""

PRICING_CATALOG = [
    # AWS instances
    {"provider": "AWS", "instance": "c6i.2xlarge",  "vcpu": 8,  "ram_gb": 16, "storage": "100GB SSD", "price_hr": 0.34,  "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "c6i.4xlarge",  "vcpu": 16, "ram_gb": 32, "storage": "100GB SSD", "price_hr": 0.68,  "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "c6i.8xlarge",  "vcpu": 32, "ram_gb": 64, "storage": "200GB SSD", "price_hr": 1.36,  "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "r6i.2xlarge",  "vcpu": 8,  "ram_gb": 64, "storage": "100GB SSD", "price_hr": 0.504, "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "r6i.4xlarge",  "vcpu": 16, "ram_gb": 128,"storage": "200GB SSD", "price_hr": 1.008, "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "p3.2xlarge",   "vcpu": 8,  "ram_gb": 61, "storage": "100GB SSD", "price_hr": 3.06,  "gpu": True,  "network_gbps": 10.0},
    {"provider": "AWS", "instance": "m6i.xlarge",   "vcpu": 4,  "ram_gb": 16, "storage": "50GB SSD",  "price_hr": 0.192, "gpu": False, "network_gbps": 12.5},
    {"provider": "AWS", "instance": "m6i.2xlarge",  "vcpu": 8,  "ram_gb": 32, "storage": "100GB SSD", "price_hr": 0.384, "gpu": False, "network_gbps": 12.5},

    # GCP instances
    {"provider": "GCP", "instance": "c2-standard-8",  "vcpu": 8,  "ram_gb": 32, "storage": "100GB SSD", "price_hr": 0.381, "gpu": False, "network_gbps": 16.0},
    {"provider": "GCP", "instance": "c2-standard-16", "vcpu": 16, "ram_gb": 64, "storage": "200GB SSD", "price_hr": 0.762, "gpu": False, "network_gbps": 16.0},
    {"provider": "GCP", "instance": "c2-standard-30", "vcpu": 30, "ram_gb": 120,"storage": "300GB SSD", "price_hr": 1.431, "gpu": False, "network_gbps": 16.0},
    {"provider": "GCP", "instance": "n2-highmem-8",   "vcpu": 8,  "ram_gb": 64, "storage": "100GB SSD", "price_hr": 0.469, "gpu": False, "network_gbps": 16.0},
    {"provider": "GCP", "instance": "n2-standard-4",  "vcpu": 4,  "ram_gb": 16, "storage": "50GB SSD",  "price_hr": 0.194, "gpu": False, "network_gbps": 16.0},

    # Azure instances
    {"provider": "Azure", "instance": "Standard_F8s_v2",  "vcpu": 8,  "ram_gb": 16, "storage": "100GB SSD", "price_hr": 0.338, "gpu": False, "network_gbps": 12.5},
    {"provider": "Azure", "instance": "Standard_F16s_v2", "vcpu": 16, "ram_gb": 32, "storage": "200GB SSD", "price_hr": 0.676, "gpu": False, "network_gbps": 12.5},
    {"provider": "Azure", "instance": "Standard_E8s_v4",  "vcpu": 8,  "ram_gb": 64, "storage": "100GB SSD", "price_hr": 0.504, "gpu": False, "network_gbps": 12.5},
    {"provider": "Azure", "instance": "Standard_D4s_v5",  "vcpu": 4,  "ram_gb": 16, "storage": "50GB SSD",  "price_hr": 0.192, "gpu": False, "network_gbps": 12.5},
    {"provider": "Azure", "instance": "Standard_D8s_v5",  "vcpu": 8,  "ram_gb": 32, "storage": "100GB SSD", "price_hr": 0.384, "gpu": False, "network_gbps": 12.5},
]

# Workflow performance traces (mock historical data)
# Format: {workflow_type: [{instance, vcpu, ram_gb, runtime_hrs, success}]}
WORKFLOW_TRACES = {
    "neuron_simulation": [
        {"instance": "c6i.2xlarge",  "vcpu": 8,  "ram_gb": 16, "runtime_hrs": 2.1, "success": True,  "cpu_util": 0.88, "ram_util": 0.45},
        {"instance": "c6i.4xlarge",  "vcpu": 16, "ram_gb": 32, "runtime_hrs": 1.1, "success": True,  "cpu_util": 0.91, "ram_util": 0.40},
        {"instance": "c2-standard-16","vcpu":16, "ram_gb": 64, "runtime_hrs": 0.9, "success": True,  "cpu_util": 0.85, "ram_util": 0.30},
        {"instance": "m6i.xlarge",   "vcpu": 4,  "ram_gb": 16, "runtime_hrs": 4.2, "success": True,  "cpu_util": 0.95, "ram_util": 0.50},
        {"instance": "r6i.2xlarge",  "vcpu": 8,  "ram_gb": 64, "runtime_hrs": 2.0, "success": True,  "cpu_util": 0.82, "ram_util": 0.20},
    ],
    "rnaseq": [
        {"instance": "r6i.4xlarge",  "vcpu": 16, "ram_gb": 128,"runtime_hrs": 3.5, "success": True,  "cpu_util": 0.70, "ram_util": 0.80},
        {"instance": "r6i.2xlarge",  "vcpu": 8,  "ram_gb": 64, "runtime_hrs": 6.2, "success": True,  "cpu_util": 0.75, "ram_util": 0.85},
        {"instance": "c6i.8xlarge",  "vcpu": 32, "ram_gb": 64, "runtime_hrs": 2.8, "success": True,  "cpu_util": 0.60, "ram_util": 0.90},
        {"instance": "n2-highmem-8", "vcpu": 8,  "ram_gb": 64, "runtime_hrs": 3.8, "success": True,  "cpu_util": 0.72, "ram_util": 0.78},
        {"instance": "Standard_E8s_v4","vcpu":8, "ram_gb": 64, "runtime_hrs": 4.0, "success": False, "cpu_util": 0.68, "ram_util": 0.92},
    ],
    "fastqc": [
        {"instance": "m6i.xlarge",   "vcpu": 4,  "ram_gb": 16, "runtime_hrs": 0.5, "success": True,  "cpu_util": 0.60, "ram_util": 0.30},
        {"instance": "c6i.2xlarge",  "vcpu": 8,  "ram_gb": 16, "runtime_hrs": 0.3, "success": True,  "cpu_util": 0.55, "ram_util": 0.25},
        {"instance": "n2-standard-4","vcpu": 4,  "ram_gb": 16, "runtime_hrs": 0.5, "success": True,  "cpu_util": 0.58, "ram_util": 0.28},
    ],
}

# Workflow stage resource profiles (for workflow decomposition)
WORKFLOW_STAGES = {
    "neuron_simulation": [
        {"stage": "preprocessing",  "cpu_intensity": "low",    "ram_intensity": "low",    "duration_fraction": 0.10},
        {"stage": "simulation",     "cpu_intensity": "high",   "ram_intensity": "medium", "duration_fraction": 0.70},
        {"stage": "postprocessing", "cpu_intensity": "medium", "ram_intensity": "medium", "duration_fraction": 0.15},
        {"stage": "visualization",  "cpu_intensity": "low",    "ram_intensity": "low",    "duration_fraction": 0.05},
    ],
    "rnaseq": [
        {"stage": "preprocessing",  "cpu_intensity": "medium", "ram_intensity": "high",   "duration_fraction": 0.20},
        {"stage": "alignment",      "cpu_intensity": "high",   "ram_intensity": "high",   "duration_fraction": 0.40},
        {"stage": "quantification", "cpu_intensity": "medium", "ram_intensity": "medium", "duration_fraction": 0.25},
        {"stage": "storage_archival","cpu_intensity":"low",    "ram_intensity": "low",    "duration_fraction": 0.15},
    ],
    "fastqc": [
        {"stage": "preprocessing",  "cpu_intensity": "low",    "ram_intensity": "low",    "duration_fraction": 0.30},
        {"stage": "quality_check",  "cpu_intensity": "medium", "ram_intensity": "low",    "duration_fraction": 0.60},
        {"stage": "storage_archival","cpu_intensity":"low",    "ram_intensity": "low",    "duration_fraction": 0.10},
    ],
}
