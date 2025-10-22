# Insulator-PowerGrid-Extractor

An automated insulator extraction algorithm from UAV LiDAR point cloud data for power transmission towers.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
- [Algorithm Pipeline](#algorithm-pipeline)
- [Citation](#citation)

## Overview

This project is an implementation of the algorithm described in the paper:

**"Insulator Extraction from UAV LiDAR Point Cloud Based on Multi-Type and Multi-Scale Feature Histogram"**
*Chen, M., Li, J., Pan, J., Ji, C., & Ma, W. (2024). Drones, 8(6), 241.*
DOI: [10.3390/drones8060241](https://doi.org/10.3390/drones8060241)

The algorithm automatically extracts insulator segments from 3D point cloud data captured by UAV-mounted LiDAR sensors using multi-type and multi-scale feature histogram analysis.

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
open3d>=0.13.0
matplotlib>=3.4.0
alphashape>=1.3.0
shapely>=2.0.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/lambdayin/Insulator-PowerGrid-Extractor.git
cd Insulator-PowerGrid-Extractor

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Process a single tower with visualization
python main.py --data-path ./data --tower-id 001 --visualize

# Process all towers and save results
python main.py --data-path ./data --plot --output-dir ./output

# Batch process without visualization (faster)
python main.py --data-path ./data --no-visualize
```

## Data Format

### Directory Structure
```
data/
├── 001Tower.txt  # Tower point cloud for tower 001
├── 001Line.txt   # Power line point cloud for tower 001
├── 002Tower.txt
├── 002Line.txt
└── ...
```

### File Format
Point cloud files should be text files with comma-separated or space-separated XYZ coordinates:
```
x1,y1,z1
x2,y2,z2
x3,y3,z3
...
```

**Note**: The algorithm expects pre-separated tower and line point clouds. Raw UAV LiDAR data needs to be segmented first.

## Usage

### Basic Usage

```bash
# Process all towers in the data directory
python main.py --data-path ./data

# Process a specific tower
python main.py --data-path ./data --tower-id 001

# Process with visualization
python main.py --data-path ./data --tower-id 001 --visualize

# Save results as PCD file
python main.py --data-path ./data --tower-id 001 --plot
```

### Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--tower-id` | str | Specific tower ID to process (e.g., 001, 002). If not specified, processes all towers |
| `--data-path` | str | Path to data directory containing Tower.txt and Line.txt files |
| `--visualize` | flag | Enable matplotlib visualization |
| `--no-visualize` | flag | Disable visualization |
| `--plot` | flag | Save point cloud as PCD file with color-coded insulators |
| `--output-dir` | str | Output directory for PCD files (default: ./output) |

## Algorithm Pipeline

The implementation follows the methodology from the paper:

1. **Data Loading**: Read tower and power line point clouds
2. **Tower Redirection**: Align tower to canonical orientation
3. **Pylon Classification**: Use VV (Vertical Void) histogram to categorize into suspension/tension types
4. **Crossarm Localization**: Apply HD (Horizontal Density) histogram to locate tower crossarms
5. **Multi-Scale Processing**: Process with 11 different grid scales (0.05m - 0.15m)
6. **Insulator Segmentation**: Extract insulators using HV (Horizontal Void) and HD histograms
7. **Adaptive Scale Selection**: Select optimal scale based on tower design standards
8. **Post-processing**: Remove outliers and duplicate points
9. **Visualization/Export**: Display or save results

The algorithm implements five histogram features: HD, HV, HW, VV, VW for multi-scale insulator extraction.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{chen2024insulator,
  title={Insulator Extraction from UAV LiDAR Point Cloud Based on Multi-Type and Multi-Scale Feature Histogram},
  author={Chen, Maolin and Li, Jiyang and Pan, Jianping and Ji, Cuicui and Ma, Wei},
  journal={Drones},
  volume={8},
  number={6},
  pages={241},
  year={2024},
  publisher={MDPI},
  doi={10.3390/drones8060241},
  url={https://doi.org/10.3390/drones8060241}
}
```