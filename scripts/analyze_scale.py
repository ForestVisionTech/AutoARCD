#!/usr/bin/env python3
"""
Analyze Scale from AprilTag Marker Detections

Triangulates marker centers from multiple images and calculates the scale factor
between reconstruction units and real-world millimeters.

Usage:
    python analyze_scale.py --sparse_path /path/to/sparse --markers_path /path/to/markers
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict


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
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_txt(path: str) -> dict:
    """Read COLMAP images.txt file."""
    images = {}
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
            name = parts[10] if len(parts) > 10 else parts[9]

            images[name] = {
                'image_id': image_id,
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
                'camera_id': camera_id
            }
            i += 2  # Skip points2D line
        else:
            i += 1

    return images


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def get_camera_matrix(camera: dict) -> np.ndarray:
    """Get camera intrinsic matrix."""
    model = camera['model']
    params = camera['params']

    if model in ('OPENCV', 'PINHOLE'):
        fx, fy, cx, cy = params[:4]
    elif model == 'SIMPLE_RADIAL':
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise ValueError(f"Unknown camera model: {model}")

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def read_marker_file(path: str) -> dict:
    """
    Read marker detection file.

    Returns:
        dict: {marker_id: {'center_2d': [x, y], 'center_3d': [x, y, z]}}
    """
    markers = {}

    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
        for marker_id, info in data.get('markers', {}).items():
            markers[int(marker_id)] = {
                'center_2d': np.array(info['center_2d']),
                'center_3d': np.array(info['center_3d'])
            }
    else:
        # TXT format
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    marker_id = int(parts[0])
                    markers[marker_id] = {
                        'center_2d': np.array([float(parts[1]), float(parts[2])]),
                        'center_3d': np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                    }

    return markers


def triangulate_point(observations: list, cameras: dict, images: dict) -> np.ndarray:
    """
    Triangulate a 3D point from multiple 2D observations using DLT.

    Args:
        observations: List of (image_name, 2d_point) tuples
        cameras: Camera intrinsics dict
        images: Image poses dict

    Returns:
        3D point or None if triangulation fails
    """
    if len(observations) < 2:
        return None

    A = []
    for img_name, pt2d in observations:
        if img_name not in images:
            continue

        img = images[img_name]
        cam = cameras[img['camera_id']]
        K = get_camera_matrix(cam)
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']

        P = K @ np.hstack([R, t.reshape(3, 1)])
        x, y = pt2d

        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    if len(A) < 4:
        return None

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]

    return X[:3]


def main():
    parser = argparse.ArgumentParser(description="Analyze scale from marker detections")
    parser.add_argument("--sparse_path", required=True,
                        help="Path to COLMAP sparse reconstruction")
    parser.add_argument("--markers_path", required=True,
                        help="Path to marker detection files")
    parser.add_argument("--output_ply", help="Output PLY file for marker centers")

    args = parser.parse_args()

    # Read reconstruction
    print("Reading reconstruction...")
    cameras = read_cameras_txt(os.path.join(args.sparse_path, 'cameras.txt'))
    images = read_images_txt(os.path.join(args.sparse_path, 'images.txt'))
    print(f"  Cameras: {len(cameras)}, Images: {len(images)}")

    # Read marker detections
    print("\nReading marker detections...")
    observations = defaultdict(list)  # marker_id -> [(img_name, center_2d), ...]
    template_3d = {}  # marker_id -> template 3D position

    # Find marker files
    for root, dirs, files in os.walk(args.markers_path):
        for filename in files:
            if not (filename.endswith('.txt') or filename.endswith('.json')):
                continue
            if filename == 'board_definition.json':
                continue

            # Determine image name in reconstruction
            rel_path = os.path.relpath(root, args.markers_path)
            img_base = os.path.splitext(filename)[0] + '.JPG'
            if rel_path != '.':
                img_name = f"{rel_path}/{img_base}"
            else:
                img_name = img_base

            # Read markers
            marker_file = os.path.join(root, filename)
            markers = read_marker_file(marker_file)

            for marker_id, info in markers.items():
                observations[marker_id].append((img_name, info['center_2d']))
                if marker_id not in template_3d:
                    template_3d[marker_id] = info['center_3d']

    print(f"  Found {len(observations)} markers:")
    for marker_id, obs in sorted(observations.items()):
        print(f"    ID {marker_id}: {len(obs)} observations")

    # Triangulate marker centers
    print("\nTriangulating marker centers...")
    centers_3d = {}

    for marker_id, obs in observations.items():
        pt3d = triangulate_point(obs, cameras, images)
        if pt3d is not None:
            centers_3d[marker_id] = pt3d
            print(f"  ID {marker_id}: ({pt3d[0]:.4f}, {pt3d[1]:.4f}, {pt3d[2]:.4f})")

    if len(centers_3d) < 2:
        print("\nERROR: Need at least 2 markers for scale analysis")
        return

    # Calculate distances
    print("\nPairwise distances (reconstruction units):")
    marker_ids = sorted(centers_3d.keys())
    distances = []

    for i, id1 in enumerate(marker_ids):
        for id2 in marker_ids[i+1:]:
            dist_recon = np.linalg.norm(centers_3d[id1] - centers_3d[id2])
            dist_template = np.linalg.norm(template_3d[id1] - template_3d[id2])
            distances.append({
                'ids': (id1, id2),
                'recon': dist_recon,
                'template': dist_template
            })
            print(f"  {id1} <-> {id2}: {dist_recon:.4f} units (expected: {dist_template:.1f}mm)")

    # Calculate scale factor
    print("\n" + "="*50)
    print("SCALE ANALYSIS")
    print("="*50)

    # Use the shortest distance (adjacent markers = 105mm)
    adjacent_distances = [d for d in distances if d['template'] < 110]  # Adjacent = 105mm
    adjacent_dist = min(adjacent_distances, key=lambda d: d['recon'])
    scale_factor = adjacent_dist['template'] / adjacent_dist['recon']

    print(f"\nReference distance: {adjacent_dist['ids'][0]} <-> {adjacent_dist['ids'][1]}")
    print(f"  Reconstruction: {adjacent_dist['recon']:.4f} units")
    print(f"  Expected: {adjacent_dist['template']:.1f} mm")
    print(f"\nScale factor: {scale_factor:.4f}")
    print(f"  1 reconstruction unit = {scale_factor:.2f} mm")
    print(f"  1 mm = {1/scale_factor:.6f} reconstruction units")

    # Verify with all distances
    print("\nVerification (all distances):")
    for d in distances:
        scaled = d['recon'] * scale_factor
        error = abs(scaled - d['template'])
        print(f"  {d['ids'][0]} <-> {d['ids'][1]}: {scaled:.1f}mm (expected {d['template']:.1f}mm, error: {error:.1f}mm)")

    # Save marker centers as PLY
    if args.output_ply:
        print(f"\nSaving marker centers to {args.output_ply}")
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        with open(args.output_ply, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(centers_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i, (marker_id, pt) in enumerate(sorted(centers_3d.items())):
                color = colors[i % len(colors)]
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")

    # Save scale info
    scale_file = os.path.join(os.path.dirname(args.output_ply) if args.output_ply else '.', 'scale_info.json')
    with open(scale_file, 'w') as f:
        json.dump({
            'scale_factor': scale_factor,
            'units': 'mm per reconstruction unit',
            'marker_centers_3d': {str(k): v.tolist() for k, v in centers_3d.items()},
            'template_positions': {str(k): v.tolist() for k, v in template_3d.items()}
        }, f, indent=2)
    print(f"Scale info saved to: {scale_file}")


if __name__ == '__main__':
    main()
