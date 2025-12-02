#!/usr/bin/env python3
"""
Dense Reconstruction using OpenMVS

Creates a dense point cloud from COLMAP sparse reconstruction.

This version bypasses COLMAP's image_undistorter (which has issues with custom COLMAP)
and directly prepares the data for OpenMVS.

Pipeline:
1. Prepare directory structure with original images
2. Convert COLMAP TXT format to standard binary format
3. Convert to OpenMVS format (InterfaceCOLMAP)
4. Dense point cloud generation (DensifyPointCloud)

Usage:
    python dense_reconstruction.py \
        --sparse_dir /path/to/sparse/0 \
        --images_dir /path/to/images \
        --output_dir /path/to/dense
"""

import argparse
import logging
import subprocess
import sys
import os
import shutil
import struct
from pathlib import Path
from collections import OrderedDict


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def run_command(cmd: list, description: str, timeout: int = 1800) -> bool:
    """Run a command with error handling and logging"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Step: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info('='*60)

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout)
        if result.stdout:
            logging.info(result.stdout)
        logging.info(f"✓ {description} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logging.error(f"✗ {description} timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ {description} failed with return code {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logging.error(f"STDERR:\n{e.stderr}")
        return False


def read_cameras_txt(path: str) -> dict:
    """Read COLMAP cameras.txt file."""
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                'id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_txt(path: str) -> dict:
    """Read COLMAP images.txt file (custom format with TIMESTAMP)."""
    images = OrderedDict()
    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = [float(parts[j]) for j in range(1, 5)]
            tx, ty, tz = [float(parts[j]) for j in range(5, 8)]
            camera_id = int(parts[8])
            # Custom COLMAP has TIMESTAMP at index 9, NAME at index 10
            # Standard COLMAP has NAME at index 9
            if len(parts) > 10:
                name = parts[10]  # Custom format with TIMESTAMP
            else:
                name = parts[9]   # Standard format

            images[image_id] = {
                'id': image_id,
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': name
            }
            i += 2  # Skip points2D line
        else:
            i += 1

    return images


def read_points3D_txt(path: str) -> dict:
    """Read COLMAP points3D.txt file."""
    points = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                xyz = [float(parts[j]) for j in range(1, 4)]
                rgb = [int(parts[j]) for j in range(4, 7)]
                error = float(parts[7])
                # Track data is pairs of (image_id, point2d_idx)
                track = []
                for j in range(8, len(parts), 2):
                    if j + 1 < len(parts):
                        track.append((int(parts[j]), int(parts[j+1])))
                points[point_id] = {
                    'id': point_id,
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'track': track
                }
    return points


def write_cameras_bin(cameras: dict, path: str, force_pinhole: bool = False):
    """Write cameras in standard COLMAP binary format.

    Args:
        cameras: Camera dictionary
        path: Output file path
        force_pinhole: If True, convert OPENCV/SIMPLE_RADIAL to PINHOLE (drops distortion)
    """
    CAMERA_MODEL_IDS = {
        'SIMPLE_PINHOLE': 0, 'PINHOLE': 1, 'SIMPLE_RADIAL': 2, 'RADIAL': 3,
        'OPENCV': 4, 'OPENCV_FISHEYE': 5, 'FULL_OPENCV': 6, 'FOV': 7,
        'SIMPLE_RADIAL_FISHEYE': 8, 'RADIAL_FISHEYE': 9, 'THIN_PRISM_FISHEYE': 10
    }

    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for camera_id, cam in cameras.items():
            model = cam['model']
            params = cam['params']

            # Convert to PINHOLE if requested (for OpenMVS compatibility)
            if force_pinhole and model in ('OPENCV', 'SIMPLE_RADIAL', 'RADIAL'):
                if model == 'OPENCV':
                    # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                    # PINHOLE: fx, fy, cx, cy
                    params = params[:4]
                elif model in ('SIMPLE_RADIAL', 'RADIAL'):
                    # SIMPLE_RADIAL: f, cx, cy, k
                    # RADIAL: f, cx, cy, k1, k2
                    # Convert to PINHOLE: fx, fy, cx, cy
                    f_val = params[0]
                    params = [f_val, f_val, params[1], params[2]]
                model = 'PINHOLE'
                logging.info(f"  Converted camera {camera_id} to PINHOLE model")

            model_id = CAMERA_MODEL_IDS.get(model, 4)  # Default to OPENCV
            f.write(struct.pack('<I', camera_id))
            f.write(struct.pack('<i', model_id))
            f.write(struct.pack('<Q', cam['width']))
            f.write(struct.pack('<Q', cam['height']))
            for param in params:
                f.write(struct.pack('<d', param))


def write_images_bin(images: dict, path: str):
    """Write images in standard COLMAP binary format (without TIMESTAMP)."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for image_id, img in images.items():
            f.write(struct.pack('<I', image_id))
            # Quaternion (qw, qx, qy, qz)
            for q in img['qvec']:
                f.write(struct.pack('<d', q))
            # Translation (tx, ty, tz)
            for t in img['tvec']:
                f.write(struct.pack('<d', t))
            f.write(struct.pack('<I', img['camera_id']))
            # Image name (null-terminated string)
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            # Empty points2D (we'll write 0 points)
            f.write(struct.pack('<Q', 0))


def write_points3D_bin(points: dict, path: str):
    """Write points3D in standard COLMAP binary format."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for point_id, pt in points.items():
            f.write(struct.pack('<Q', point_id))
            # XYZ
            for coord in pt['xyz']:
                f.write(struct.pack('<d', coord))
            # RGB
            for c in pt['rgb']:
                f.write(struct.pack('<B', c))
            # Error
            f.write(struct.pack('<d', pt['error']))
            # Track length and data
            f.write(struct.pack('<Q', len(pt['track'])))
            for img_id, pt2d_idx in pt['track']:
                f.write(struct.pack('<I', img_id))
                f.write(struct.pack('<I', pt2d_idx))


def convert_txt_to_standard_bin(sparse_txt_dir: Path, output_bin_dir: Path):
    """Convert COLMAP TXT format to standard binary format."""
    logging.info("Converting COLMAP TXT to standard binary format...")

    output_bin_dir.mkdir(parents=True, exist_ok=True)

    # Read TXT files
    cameras = read_cameras_txt(sparse_txt_dir / 'cameras.txt')
    images = read_images_txt(sparse_txt_dir / 'images.txt')
    points3D = read_points3D_txt(sparse_txt_dir / 'points3D.txt')

    logging.info(f"  Read {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")

    # Write standard binary files (force PINHOLE for OpenMVS compatibility)
    write_cameras_bin(cameras, output_bin_dir / 'cameras.bin', force_pinhole=True)
    write_images_bin(images, output_bin_dir / 'images.bin')
    write_points3D_bin(points3D, output_bin_dir / 'points3D.bin')

    logging.info(f"  Written standard binary files to {output_bin_dir}")

    return cameras, images, points3D


def setup_openmvs_structure(images_dir: Path, output_dir: Path, images: dict):
    """Set up directory structure for OpenMVS."""
    logging.info("Setting up OpenMVS directory structure...")

    # Create images directory
    openmvs_images_dir = output_dir / "images"
    openmvs_images_dir.mkdir(parents=True, exist_ok=True)

    # Find and copy/symlink images
    copied = 0
    for img_id, img_info in images.items():
        img_name = img_info['name']
        src_path = images_dir / img_name

        if src_path.exists():
            # Get just the filename without subdirectory
            dst_name = Path(img_name).name
            dst_path = openmvs_images_dir / dst_name

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

    logging.info(f"  Copied {copied} images to {openmvs_images_dir}")
    return openmvs_images_dir


def update_image_names_for_openmvs(images: dict) -> dict:
    """Update image names to remove subdirectory path (OpenMVS expects flat structure)."""
    updated = OrderedDict()
    for img_id, img_info in images.items():
        new_info = img_info.copy()
        # Remove subdirectory from path (e.g., "images/IMG_9023.JPG" -> "IMG_9023.JPG")
        new_info['name'] = Path(img_info['name']).name
        updated[img_id] = new_info
    return updated


def main():
    parser = argparse.ArgumentParser(description="Run OpenMVS dense reconstruction")
    parser.add_argument("--sparse_dir", required=True,
                        help="Path to COLMAP sparse reconstruction (contains cameras.txt, images.txt, points3D.txt)")
    parser.add_argument("--images_dir", required=True,
                        help="Path to images directory (parent of subdirectory containing images)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for dense reconstruction")

    # OpenMVS parameters
    parser.add_argument("--resolution_level", type=int, default=1,
                        help="Resolution level (default: 1, higher = faster but less detail)")
    parser.add_argument("--number_views", type=int, default=5,
                        help="Number of views for dense reconstruction (default: 5)")

    args = parser.parse_args()

    setup_logging()

    # Convert to Path objects
    sparse_dir = Path(args.sparse_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    logging.info("OpenMVS Dense Reconstruction (Direct Method)")
    logging.info(f"Sparse directory: {sparse_dir}")
    logging.info(f"Images directory: {images_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if sparse reconstruction exists (TXT format)
    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in required_files:
        if not (sparse_dir / f).exists():
            logging.error(f"Required file not found: {sparse_dir / f}")
            return 1

    # Step 1: Read TXT and convert to standard binary format
    logging.info("\n[Step 1/4] Converting COLMAP format...")
    # InterfaceCOLMAP expects: input_dir/sparse/cameras.bin, etc.
    standard_sparse_dir = output_dir / "colmap" / "sparse"
    cameras, images, points3D = convert_txt_to_standard_bin(sparse_dir, standard_sparse_dir)

    # Step 2: Set up images for OpenMVS (flat structure)
    logging.info("\n[Step 2/4] Setting up images...")
    openmvs_images_dir = setup_openmvs_structure(images_dir, output_dir, images)

    # Update image names in sparse model for flat structure
    images_flat = update_image_names_for_openmvs(images)
    write_images_bin(images_flat, standard_sparse_dir / 'images.bin')

    # Step 3: Convert to OpenMVS format
    logging.info("\n[Step 3/4] Converting to OpenMVS format...")
    scene_file = output_dir / "scene.mvs"

    # InterfaceCOLMAP expects input-file to be the parent of 'sparse' directory
    colmap_dir = output_dir / "colmap"

    cmd2 = [
        "InterfaceCOLMAP",
        "--working-folder", str(output_dir),
        "--output-file", str(scene_file),
        "--input-file", str(colmap_dir),
        "--image-folder", str(openmvs_images_dir)
    ]

    if not run_command(cmd2, "InterfaceCOLMAP conversion"):
        return 1

    # Step 4: Dense point cloud generation
    logging.info("\n[Step 4/4] Dense Point Cloud Generation...")
    dense_file = output_dir / "dense.mvs"

    cmd3 = [
        "DensifyPointCloud",
        "--input-file", str(scene_file),
        "--output-file", str(dense_file),
        "--resolution-level", str(args.resolution_level),
        "--number-views", str(args.number_views),
        "--estimate-colors", "1",
        "--working-folder", str(output_dir)
    ]

    if not run_command(cmd3, "DensifyPointCloud", timeout=3600):
        return 1

    # Check for output files
    dense_ply = output_dir / "dense.ply"

    logging.info("\n" + "="*60)
    logging.info("Dense Reconstruction Complete!")
    logging.info("="*60)

    if dense_ply.exists():
        ply_size = dense_ply.stat().st_size / (1024*1024)
        logging.info(f"Dense PLY: {dense_ply} ({ply_size:.1f} MB)")
    else:
        logging.warning(f"Dense PLY not found at expected location: {dense_ply}")

    if scene_file.exists():
        logging.info(f"Scene file: {scene_file}")

    if dense_file.exists():
        logging.info(f"Dense MVS: {dense_file}")

    logging.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
