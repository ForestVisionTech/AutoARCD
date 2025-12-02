# ARCD - AprilTag Reconstruction with COLMAP for Dimensions

3D reconstruction from iPhone photos with real-world scale using AprilTag markers.

## Docker Workflow

### Prerequisites

- Docker with GPU support (NVIDIA)
- Custom COLMAP Docker image with Shape3D support

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
cd /workspace2/arcd
./run_pipeline.sh /workspace2/your_images project_name
```

**Option 2: Step-by-step execution**
```bash
# 1. Detect AprilTag markers
python3 scripts/detect_markers.py \
    --image_path /workspace2/your_images/images \
    --output_path /workspace2/arcd/projects/myproject/markers/images \
    --center_distance 105.0 \
    --marker_ids "99,352,1676,2067" \
    --visualize

# 2. Run COLMAP feature extraction
colmap feature_extractor \
    --database_path /workspace2/arcd/projects/myproject/database.db \
    --image_path /workspace2/your_images \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1

# 3. Run COLMAP feature matching
colmap exhaustive_matcher \
    --database_path /workspace2/arcd/projects/myproject/database.db \
    --SiftMatching.use_gpu 1

# 4. Run COLMAP sparse reconstruction
colmap mapper \
    --database_path /workspace2/arcd/projects/myproject/database.db \
    --image_path /workspace2/your_images \
    --output_path /workspace2/arcd/projects/myproject/sparse \
    --Mapper.ba_use_gpu 1

# 5. Export to PLY
colmap model_converter \
    --input_path /workspace2/arcd/projects/myproject/sparse/0 \
    --output_path /workspace2/arcd/projects/myproject/sparse/0 \
    --output_type TXT

colmap model_converter \
    --input_path /workspace2/arcd/projects/myproject/sparse/0 \
    --output_path /workspace2/arcd/projects/myproject/output/reconstruction.ply \
    --output_type PLY

# 6. Localize board and compute scale
python3 scripts/localize_board_final.py \
    --sparse_path /workspace2/arcd/projects/myproject/sparse/0 \
    --markers_path /workspace2/arcd/projects/myproject/markers \
    --output_ply /workspace2/arcd/projects/myproject/output/marker_centers.ply

# 7. Combine point clouds (optional, done automatically by step 6)
python3 scripts/combine_pointclouds.py \
    --reconstruction_ply /workspace2/arcd/projects/myproject/output/reconstruction.ply \
    --markers_ply /workspace2/arcd/projects/myproject/output/marker_centers.ply \
    --output_ply /workspace2/arcd/projects/myproject/output/combined_with_markers.ply \
    --marker_radius 0.03
```

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
    |   [99]----[352]
    |     |       |      105mm
    |  [1676]--[2067]
    +-------> +X

    Origin at marker 2067
```

## Project Structure

```
arcd/
├── README.md                     # This file
├── run_pipeline.sh               # Automated pipeline script
├── scripts/
│   ├── detect_markers.py         # AprilTag detection
│   ├── localize_board_final.py   # PnP-based board localization
│   ├── analyze_scale.py          # Triangulation-based scale (legacy)
│   ├── combine_pointclouds.py    # Merge reconstruction with markers
│   └── dense_reconstruction.py   # OpenMVS dense reconstruction
└── projects/                     # Output projects
```

## Output Files

After running the pipeline, you'll find:

```
projects/myproject/
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

## Important Notes

1. **Keep the board stationary**: For best results, the ChArUco board should not move during image capture. The board's position is used to establish real-world scale.

2. **Image directory structure**: COLMAP expects images in a subdirectory:
   ```
   your_images/
   └── images/
       ├── IMG_0001.JPG
       ├── IMG_0002.JPG
       └── ...
   ```

3. **Marker visibility**: For accurate scale, ensure all 4 markers are visible in multiple images. More observations = better accuracy.

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No markers detected | Check marker IDs match your board |
| Poor geometry (edges unequal) | Board may have moved during capture |
| Markers not on board in visualization | Re-run with stationary board |
| COLMAP reconstruction fails | Ensure sufficient image overlap |
