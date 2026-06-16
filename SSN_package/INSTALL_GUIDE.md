# SSN Package Installation Guide

## Package Information
- **Package Name**: SSN (Sample-Specific Network)
- **Version**: 0.0.0.9000
- **Purpose**: Sample-specific network analysis for microbial communities

## Installation

### Method 1: Install from local directory
```r
devtools::install_local("E:/Onedrive/labZhouc/Paper_Conc/Final0203V2/SSN_package")
```

### Method 2: Load for development
```r
devtools::load_all("E:/Onedrive/labZhouc/Paper_Conc/Final0203V2/SSN_package")
```

## Main Functions

1. **generate_SSN()**: Generate sample-specific networks from phyloseq data
2. **extr_netattr()**: Extract network attributes
3. **extr_degree()**: Extract degree information
4. **generate_edgelist()**: Generate edge lists
5. **generate_samplespecificabundance()**: Generate sample-specific abundance
6. **create_phyloseq_object()**: Create phyloseq objects

## Usage Example

```r
library(SSN)

# Set your data path
mypath <- "path/to/your/data/"

# Generate SSN
generate_SSN(
  mypath = mypath,
  phylo_list = c('your_phyloseq_data.Rdata')
)
```

## Notes
- All references to "NetDiverse" have been replaced with "SSN"
- Package structure follows standard R package conventions
- Tutorial and documentation are available in the package directory
