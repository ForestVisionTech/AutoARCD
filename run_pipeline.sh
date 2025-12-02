#!/bin/bash
#
# AprilTag-based 3D Reconstruction Pipeline
#
# Usage: ./run_pipeline.sh <images_folder> [project_name]
#
# Example:
#   ./run_pipeline.sh /workspace2/11-08-log016back log016
#
# Requirements:
#   - Custom COLMAP with Shape3D support
#   - Python 3 with opencv-contrib-python, numpy
#   - Images must be in a subdirectory (e.g., images_folder/images/*.JPG)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Board configuration
# Marker IDs in order: top-left, top-right, bottom-left, bottom-right
CENTER_DISTANCE=105.0
MARKER_IDS="99,2067,352,1676"

# Parse arguments
IMAGES_PATH="${1:?Usage: $0 <images_folder> [project_name]}"
PROJECT_NAME="${2:-$(basename "$IMAGES_PATH")_$(date +%Y%m%d_%H%M%S)}"
PROJECT_PATH="${SCRIPT_DIR}/projects/${PROJECT_NAME}"

echo "============================================================"
echo "  AprilTag 3D Reconstruction Pipeline"
echo "============================================================"
echo ""
echo "Images:  $IMAGES_PATH"
echo "Project: $PROJECT_PATH"
echo "Board:   AprilTag 36h10, ${CENTER_DISTANCE}mm spacing"
echo "Markers: $MARKER_IDS"
echo ""

# Check directory structure
if [ -z "$(find "$IMAGES_PATH" -mindepth 2 -maxdepth 2 \( -name '*.JPG' -o -name '*.jpg' \) 2>/dev/null | head -1)" ]; then
    echo "Moving images to subdirectory..."
    mkdir -p "$IMAGES_PATH/images"
    mv "$IMAGES_PATH"/*.JPG "$IMAGES_PATH/images/" 2>/dev/null || true
    mv "$IMAGES_PATH"/*.jpg "$IMAGES_PATH/images/" 2>/dev/null || true
fi

IMAGES_SUBDIR=$(find "$IMAGES_PATH" -mindepth 1 -maxdepth 1 -type d | head -1)
SUBDIR_NAME=$(basename "$IMAGES_SUBDIR")

mkdir -p "$PROJECT_PATH"/{markers/$SUBDIR_NAME,sparse,output}

# Step 1: Detect markers
echo ""
echo "[Step 1/6] Detecting AprilTag markers..."
python3 "${SCRIPT_DIR}/scripts/detect_markers.py" \
    --image_path "$IMAGES_SUBDIR" \
    --output_path "$PROJECT_PATH/markers/$SUBDIR_NAME" \
    --center_distance "$CENTER_DISTANCE" \
    --marker_ids "$MARKER_IDS" \
    --visualize

# Step 2: Extract features
echo ""
echo "[Step 2/6] Extracting SIFT features..."
colmap feature_extractor \
    --database_path "$PROJECT_PATH/database.db" \
    --image_path "$IMAGES_PATH" \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1

# Step 3: Match features
echo ""
echo "[Step 3/6] Matching features..."
colmap exhaustive_matcher \
    --database_path "$PROJECT_PATH/database.db" \
    --SiftMatching.use_gpu 1

# Step 4: Sparse reconstruction
echo ""
echo "[Step 4/6] Running sparse reconstruction..."
colmap mapper \
    --database_path "$PROJECT_PATH/database.db" \
    --image_path "$IMAGES_PATH" \
    --output_path "$PROJECT_PATH/sparse" \
    --Mapper.ba_use_gpu 1 \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_refine_extra_params 1

# Find best reconstruction
BEST_RECON=""
BEST_SIZE=0
for recon_dir in "$PROJECT_PATH/sparse"/*/; do
    if [ -f "$recon_dir/images.bin" ]; then
        size=$(stat -c%s "$recon_dir/images.bin" 2>/dev/null || stat -f%z "$recon_dir/images.bin" 2>/dev/null)
        if [ "$size" -gt "$BEST_SIZE" ]; then
            BEST_SIZE=$size
            BEST_RECON="${recon_dir%/}"
        fi
    fi
done

if [ -z "$BEST_RECON" ]; then
    echo "ERROR: No reconstruction found"
    exit 1
fi

echo "Using reconstruction: $BEST_RECON"

# Step 5: Colorize and export
echo ""
echo "[Step 5/6] Colorizing and exporting..."

COLORED_PATH="${BEST_RECON}_colored"
mkdir -p "$COLORED_PATH"

colmap color_extractor \
    --image_path "$IMAGES_PATH" \
    --input_path "$BEST_RECON" \
    --output_path "$COLORED_PATH"

colmap model_converter \
    --input_path "$BEST_RECON" \
    --output_path "$BEST_RECON" \
    --output_type TXT

colmap model_converter \
    --input_path "$COLORED_PATH" \
    --output_path "$PROJECT_PATH/output/reconstruction.ply" \
    --output_type PLY

# Step 6: Localize board and analyze scale
echo ""
echo "[Step 6/7] Localizing board and analyzing scale..."
python3 "${SCRIPT_DIR}/scripts/localize_board_final.py" \
    --sparse_path "$BEST_RECON" \
    --markers_path "$PROJECT_PATH/markers" \
    --output_ply "$PROJECT_PATH/output/marker_centers.ply"

# Note: localize_board_final.py automatically calls combine_pointclouds.py
# But we run it again here to ensure the combined file exists
echo ""
echo "[Step 7/7] Combining reconstruction with marker centers..."
python3 "${SCRIPT_DIR}/scripts/combine_pointclouds.py" \
    --reconstruction_ply "$PROJECT_PATH/output/reconstruction.ply" \
    --markers_ply "$PROJECT_PATH/output/marker_centers.ply" \
    --output_ply "$PROJECT_PATH/output/combined_with_markers.ply" \
    --marker_radius 0.03

# Summary
echo ""
echo "============================================================"
echo "  SPARSE RECONSTRUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Point cloud:        $PROJECT_PATH/output/reconstruction.ply"
echo "  With markers:       $PROJECT_PATH/output/combined_with_markers.ply"
echo "  Marker centers:     $PROJECT_PATH/output/marker_centers.ply"
echo "  Scale info:         $PROJECT_PATH/output/scale_info.json"
echo "  Sparse model:       $BEST_RECON"
echo ""
echo "Camera:"
grep -v "^#" "$BEST_RECON/cameras.txt"
echo ""

# Optional: Dense reconstruction
if [ "${SKIP_DENSE:-0}" != "1" ]; then
    echo ""
    echo "============================================================"
    echo "  DENSE RECONSTRUCTION (Optional)"
    echo "============================================================"
    echo ""
    echo "To run dense reconstruction:"
    echo "  python3 ${SCRIPT_DIR}/scripts/dense_reconstruction.py \\"
    echo "      --sparse_dir $BEST_RECON \\"
    echo "      --images_dir $IMAGES_PATH \\"
    echo "      --output_dir $PROJECT_PATH/dense"
    echo ""
    echo "Or set SKIP_DENSE=0 and re-run to include dense reconstruction"
fi
