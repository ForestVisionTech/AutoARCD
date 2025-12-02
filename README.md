# AutoARCD - AprilTag Reconstruction with COLMAP for Dimensions

3D reconstruction from iPhone photos with real-world scale using AprilTag markers.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Data](#test-data)
- [Input Requirements](#input-requirements)
- [Docker Workflow](#docker-workflow)
  - [Prerequisites](#prerequisites)
  - [Starting the Docker Container](#starting-the-docker-container)
  - [Running the Pipeline](#running-the-pipeline)
- [Board Configuration](#board-configuration)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Scale Information](#scale-information)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Quick Start

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/ForestVisionTech/AutoARCD.git

# 2. Download test data from S3
aws s3 sync s3://vision-computation-s3-dev/auto-arcd-data/11-08-log012-front ./data/11-08-log012-front

# 3. Run in Docker container
docker exec -it colmap_container bash
cd /workspace2/AutoARCD
./run_pipeline.sh /workspace2/data/11-08-log012-front log012
```

---

## Test Data

Sample datasets are available on S3 for testing and development.

### S3 Location

```
s3://vision-computation-s3-dev/auto-arcd-data/
├── 11-08-log012-front/     # 80 images, front view of log 012
│   └── images/
│       └── IMG_8417.JPG ... IMG_8496.JPG
└── 11-08-log016-back/      # 109 images, back view of log 016
    └── images/
        └── IMG_9023.JPG ... IMG_9131.JPG
```

### Download Commands

```bash
# Download specific dataset
aws s3 sync s3://vision-computation-s3-dev/auto-arcd-data/11-08-log012-front ./data/11-08-log012-front
aws s3 sync s3://vision-computation-s3-dev/auto-arcd-data/11-08-log016-back ./data/11-08-log016-back

# Download all test data
aws s3 sync s3://vision-computation-s3-dev/auto-arcd-data/ ./data/
```

### Running with Test Data

```bash
# Inside Docker container
./run_pipeline.sh /workspace2/data/11-08-log012-front log012-front
./run_pipeline.sh /workspace2/data/11-08-log016-back log016-back
```

---

## Input Requirements

The pipeline expects a specific input folder structure:

### Required Structure

```
your_dataset/
└── images/
    ├── IMG_0001.JPG
    ├── IMG_0002.JPG
    ├── IMG_0003.JPG
    └── ...
```

### Requirements

| Requirement | Details |
|-------------|---------|
| **Folder structure** | Images MUST be in a subdirectory named `images/` |
| **Image format** | JPG/JPEG (iPhone photos work best) |
| **Naming convention** | Any consistent naming (e.g., `IMG_XXXX.JPG`) |
| **Minimum images** | At least 20-30 images recommended |
| **Image overlap** | 60-80% overlap between consecutive images |
| **AprilTag visibility** | All 4 markers visible in multiple images |
| **Board position** | Keep the calibration board stationary during capture |

### Capture Tips

1. **Walk around the object** - Capture from multiple angles
2. **Maintain consistent distance** - Don't zoom in/out dramatically
3. **Good lighting** - Avoid harsh shadows on the AprilTag board
4. **Keep board in frame** - Include the calibration board in many images
5. **Steady shots** - Avoid motion blur

### What NOT to Do

- Don't put images directly in the root folder (must be in `images/` subdirectory)
- Don't mix different camera sources in one dataset
- Don't move the AprilTag board during capture
- Don't use heavily compressed or resized images

---

## Docker Workflow

### Prerequisites

- Docker with GPU support (NVIDIA)
- Custom COLMAP Docker image with Shape3D support
- AWS CLI (for downloading test data)

### Starting the Docker Container

```bash
# Start the container (if not already running)
docker run -d --gpus all \
    -v /path/to/data:/workspace2 \
    --name colmap_container \
    your-colmap-image:latest \
    tail -f /dev/null

# Or attach to an existing container
docker exec -it colmap_container bash
```

### Running the Pipeline

**Option 1: Full automated pipeline**
```bash
# Inside Docker container
cd /workspace2/AutoARCD
./run_pipeline.sh /workspace2/data/11-08-log012-front log012-front
```

**Option 2: Step-by-step execution**
```bash
# 1. Detect AprilTag markers
python3 scripts/detect_markers.py \
    --image_path /workspace2/data/11-08-log012-front/images \
    --output_path /workspace2/AutoARCD/projects/log012/markers/images \
    --center_distance 105.0 \
    --marker_ids "99,352,1676,2067" \
    --visualize

# 2. Run COLMAP feature extraction
colmap feature_extractor \
    --database_path /workspace2/AutoARCD/projects/log012/database.db \
    --image_path /workspace2/data/11-08-log012-front \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1

# 3. Run COLMAP feature matching
colmap exhaustive_matcher \
    --database_path /workspace2/AutoARCD/projects/log012/database.db \
    --SiftMatching.use_gpu 1

# 4. Run COLMAP sparse reconstruction
colmap mapper \
    --database_path /workspace2/AutoARCD/projects/log012/database.db \
    --image_path /workspace2/data/11-08-log012-front \
    --output_path /workspace2/AutoARCD/projects/log012/sparse \
    --Mapper.ba_use_gpu 1

# 5. Export to PLY
colmap model_converter \
    --input_path /workspace2/AutoARCD/projects/log012/sparse/0 \
    --output_path /workspace2/AutoARCD/projects/log012/sparse/0 \
    --output_type TXT

colmap model_converter \
    --input_path /workspace2/AutoARCD/projects/log012/sparse/0 \
    --output_path /workspace2/AutoARCD/projects/log012/output/reconstruction.ply \
    --output_type PLY

# 6. Localize board and compute scale
python3 scripts/localize_board_final.py \
    --sparse_path /workspace2/AutoARCD/projects/log012/sparse/0 \
    --markers_path /workspace2/AutoARCD/projects/log012/markers \
    --output_ply /workspace2/AutoARCD/projects/log012/output/marker_centers.ply

# 7. Combine point clouds (optional, done automatically by step 6)
python3 scripts/combine_pointclouds.py \
    --reconstruction_ply /workspace2/AutoARCD/projects/log012/output/reconstruction.ply \
    --markers_ply /workspace2/AutoARCD/projects/log012/output/marker_centers.ply \
    --output_ply /workspace2/AutoARCD/projects/log012/output/combined_with_markers.ply \
    --marker_radius 0.03
```

---

## Board Configuration

| Property | Value |
|----------|-------|
| AprilTag family | 36h10 |
| Marker IDs | 99, 352, 1676, 2067 |
| Layout | 2x2 grid |
| Center-to-center | 105mm |

```
Board layout (looking at board):

    +Y
    ^
    |   [99]----[2067]
    |     |       |      105mm
    |  [352]---[1676]
    +-------> +X

    Origin at marker 1676
```

---

## Project Structure

```
AutoARCD/
├── README.md                     # This file
├── run_pipeline.sh               # Automated pipeline script
├── colmap/                       # COLMAP submodule (arcd_dev branch)
├── scripts/
│   ├── detect_markers.py         # AprilTag detection
│   ├── localize_board_final.py   # PnP-based board localization
│   ├── analyze_scale.py          # Triangulation-based scale (legacy)
│   ├── combine_pointclouds.py    # Merge reconstruction with markers
│   └── dense_reconstruction.py   # OpenMVS dense reconstruction
├── docs/                         # Documentation
│   ├── marker_localization_analysis.md
│   ├── COLMAP_FV_CUSTOMIZATION_ANALYSIS.md
│   └── IPHONE_CHARUCO_RECONSTRUCTION_PLAN.md
└── projects/                     # Output projects (generated)
```

---

## Output Files

After running the pipeline, you'll find:

```
projects/log012/
├── database.db                   # COLMAP database
├── markers/images/               # Marker detection files
│   ├── IMG_XXXX.txt              # Per-image marker detections
│   └── board_definition.json     # Board geometry
├── sparse/0/                     # COLMAP sparse reconstruction
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
└── output/
    ├── reconstruction.ply        # Main point cloud
    ├── marker_centers.ply        # 4 marker center points
    ├── combined_with_markers.ply # Reconstruction + colored marker spheres
    └── scale_info.json           # Scale factor and marker positions
```

---

## Scale Information

The `scale_info.json` file contains:
- `scale_factor`: mm per COLMAP unit
- `marker_centers_3d`: 3D positions of each marker in COLMAP coordinates
- `best_image`: The image used for PnP localization
- `reproj_error_px`: Reprojection error in pixels

To convert COLMAP coordinates to millimeters:
```python
position_mm = position_colmap * scale_factor
```

---

## Known Limitations

The current marker localization has limitations. See [docs/marker_localization_analysis.md](docs/marker_localization_analysis.md) for details.

| Component | Status | Accuracy |
|-----------|--------|----------|
| COLMAP Reconstruction | Working well | 1.13 px reprojection error |
| Marker Localization | Limited accuracy | ~900 px reprojection error |
| Scale Factor | Approximate | ±15-20% uncertainty |

**Root cause**: ArUco marker center detections are not geometrically consistent across different viewing angles, even though COLMAP camera poses are accurate.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No markers detected | Check marker IDs match your board |
| Poor geometry (edges unequal) | Board may have moved during capture |
| Markers not on board in visualization | Re-run with stationary board |
| COLMAP reconstruction fails | Ensure sufficient image overlap |
| "Images must be in subdirectory" error | Move images to `your_folder/images/` |

---

## Documentation

- [Marker Localization Analysis](docs/marker_localization_analysis.md) - Detailed analysis of reconstruction accuracy
- [COLMAP FV Customization](docs/COLMAP_FV_CUSTOMIZATION_ANALYSIS.md) - Custom COLMAP modifications
- [iPhone ChArUco Reconstruction Plan](docs/IPHONE_CHARUCO_RECONSTRUCTION_PLAN.md) - Planning document
