#!/usr/bin/env python3
"""
Localize ChArUco board in COLMAP coordinates using global optimization.

This script finds the optimal board pose that minimizes reprojection error
across all views where markers are detected.
"""

import argparse
import json
import os
import numpy as np
import cv2
from scipy.optimize import minimize


def read_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }
    return cameras


def read_images_txt(path):
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
            # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME (or with TIMESTAMP before NAME)
            # Name is the last field
            name = parts[-1]
            images[name] = {
                'qvec': np.array([float(parts[j]) for j in range(1, 5)]),
                'tvec': np.array([float(parts[j]) for j in range(5, 8)]),
                'camera_id': int(parts[8])
            }
            i += 2
        else:
            i += 1
    return images


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def get_camera_intrinsics(camera):
    params = camera['params']
    if camera['model'] in ('OPENCV', 'PINHOLE'):
        fx, fy, cx, cy = params[:4]
        if camera['model'] == 'OPENCV' and len(params) >= 8:
            dist_coeffs = np.array(params[4:8])
        else:
            dist_coeffs = np.zeros(4)
    else:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3] if len(params) > 3 else 0
        dist_coeffs = np.array([k1, 0, 0, 0])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, dist_coeffs


def read_marker_file(path):
    markers = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                markers[int(parts[0])] = {
                    'center_2d': np.array([float(parts[1]), float(parts[2])]),
                    'center_3d': np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                }
    return markers


def rodrigues_to_rotmat(rvec):
    """Convert Rodrigues vector to rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotmat_to_rodrigues(R):
    """Convert rotation matrix to Rodrigues vector."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)
    k = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(theta))
    return theta * k


def project_points(points_3d, R_world, t_world, R_cam, t_cam, K, dist_coeffs):
    """Project 3D world points to 2D image coordinates."""
    # Transform from world to camera frame
    # P_cam = R_cam @ (R_world @ P_board + t_world) + t_cam
    points_world = (R_world @ points_3d.T).T + t_world
    points_cam = (R_cam @ points_world.T).T + t_cam

    # Project to image
    projected = []
    for pt in points_cam:
        if pt[2] <= 0:
            projected.append(np.array([np.inf, np.inf]))
            continue
        x = pt[0] / pt[2]
        y = pt[1] / pt[2]

        # Apply distortion
        r2 = x*x + y*y
        k1, k2, p1, p2 = dist_coeffs[:4] if len(dist_coeffs) >= 4 else [0, 0, 0, 0]
        radial = 1 + k1*r2 + k2*r2*r2
        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y

        u = K[0, 0] * x_dist + K[0, 2]
        v = K[1, 1] * y_dist + K[1, 2]
        projected.append(np.array([u, v]))

    return np.array(projected)


def compute_reprojection_error(params, observations, template_3d, images, cameras):
    """
    Compute total reprojection error for given board pose.

    params: [rx, ry, rz, tx, ty, tz, scale] - board pose in world coordinates
    """
    rvec = params[:3]
    tvec = params[3:6]
    scale = params[6]

    R_world = rodrigues_to_rotmat(rvec)
    t_world = tvec

    # Scale template
    scaled_template = {mid: pos / scale for mid, pos in template_3d.items()}

    total_error = 0.0
    n_obs = 0

    for img_name, markers in observations.items():
        if img_name not in images:
            continue

        img = images[img_name]
        cam = cameras[img['camera_id']]
        K, dist = get_camera_intrinsics(cam)
        R_cam = qvec2rotmat(img['qvec'])
        t_cam = img['tvec']

        for mid, obs in markers.items():
            if mid not in scaled_template:
                continue

            pt_3d = scaled_template[mid].reshape(1, 3)
            pt_2d_obs = obs['center_2d']

            pt_2d_proj = project_points(pt_3d, R_world, t_world, R_cam, t_cam, K, dist)[0]

            if np.any(np.isinf(pt_2d_proj)):
                total_error += 1000.0
            else:
                error = np.linalg.norm(pt_2d_proj - pt_2d_obs)
                total_error += error
            n_obs += 1

    return total_error / max(n_obs, 1)


def triangulate_marker_opencv(marker_id, all_markers, images, cameras):
    """Triangulate a marker using OpenCV's triangulatePoints with first two views."""
    pts_2d = []
    proj_mats = []

    for img_name, mkrs in all_markers.items():
        if marker_id not in mkrs:
            continue
        if img_name not in images:
            continue
        img = images[img_name]
        cam = cameras[img['camera_id']]
        K, _ = get_camera_intrinsics(cam)
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        P = K @ np.hstack([R, t.reshape(3, 1)])
        pt = mkrs[marker_id]['center_2d']
        pts_2d.append(pt)
        proj_mats.append(P)

        if len(pts_2d) >= 2:
            break

    if len(pts_2d) < 2:
        return None

    pts1 = np.array([pts_2d[0]]).T
    pts2 = np.array([pts_2d[1]]).T
    pt_4d = cv2.triangulatePoints(proj_mats[0], proj_mats[1], pts1, pts2)
    return (pt_4d[:3] / pt_4d[3]).flatten()


def pnp_estimate_pose(all_markers, template_3d, images, cameras, scale):
    """Use PnP to estimate initial board pose from a single view."""
    for img_name, markers in all_markers.items():
        if img_name not in images:
            continue
        if len(markers) < 4:
            continue

        img = images[img_name]
        cam = cameras[img['camera_id']]
        K, dist = get_camera_intrinsics(cam)
        R_cam = qvec2rotmat(img['qvec'])
        t_cam = img['tvec']

        # Collect 2D-3D correspondences (in board frame)
        obj_pts = []
        img_pts = []
        for mid, obs in markers.items():
            if mid in template_3d:
                obj_pts.append(template_3d[mid] / scale)
                img_pts.append(obs['center_2d'])

        if len(obj_pts) < 4:
            continue

        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)

        # Solve PnP to get board pose in camera frame
        success, rvec_cam, tvec_cam = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # Convert to world frame
        R_board_cam = rodrigues_to_rotmat(rvec_cam.flatten())
        t_board_cam = tvec_cam.flatten()

        # Board in world = R_cam^-1 @ (P_cam - t_cam)
        # P_cam = R_board_cam @ P_board + t_board_cam
        # P_world = R_cam^T @ (R_board_cam @ P_board + t_board_cam - t_cam)
        # = R_cam^T @ R_board_cam @ P_board + R_cam^T @ (t_board_cam - t_cam)
        R_world = R_cam.T @ R_board_cam
        t_world = R_cam.T @ (t_board_cam - t_cam)

        rvec_world = rotmat_to_rodrigues(R_world)

        return rvec_world, t_world, img_name

    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_path", required=True)
    parser.add_argument("--markers_path", required=True)
    parser.add_argument("--output_ply", required=True)
    args = parser.parse_args()

    cameras = read_cameras_txt(os.path.join(args.sparse_path, 'cameras.txt'))
    images = read_images_txt(os.path.join(args.sparse_path, 'images.txt'))

    print(f"Loaded {len(cameras)} cameras, {len(images)} images")

    # Load markers
    all_markers = {}
    template_3d = {}

    for root, dirs, files in os.walk(args.markers_path):
        for filename in files:
            if not filename.endswith('.txt') or filename == 'board_definition.json':
                continue

            rel_path = os.path.relpath(root, args.markers_path)
            img_base = os.path.splitext(filename)[0] + '.JPG'
            img_name = f"{rel_path}/{img_base}" if rel_path != '.' else img_base

            if img_name not in images:
                continue

            markers = read_marker_file(os.path.join(root, filename))
            if len(markers) == 4:
                all_markers[img_name] = markers
                for mid, info in markers.items():
                    if mid not in template_3d:
                        template_3d[mid] = info['center_3d']

    print(f"Found {len(all_markers)} images with all 4 markers")

    # Get marker IDs from template
    marker_ids = list(template_3d.keys())
    print(f"Marker IDs: {marker_ids}")
    print(f"Template 3D positions (mm):")
    for mid in marker_ids:
        pos = template_3d[mid]
        print(f"  ID {mid}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # Step 1: Triangulate to estimate scale
    print("\nStep 1: Triangulating marker positions to estimate scale...")
    triangulated = {}
    for mid in marker_ids:
        pos = triangulate_marker_opencv(mid, all_markers, images, cameras)
        if pos is not None:
            triangulated[mid] = pos
            print(f"  ID {mid}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    if len(triangulated) < 4:
        print("ERROR: Could not triangulate all 4 markers")
        return

    # Compute scale from triangulated edge distances
    print("\nComputing scale from triangulated distances...")
    edge_scales = []
    for i, id1 in enumerate(marker_ids):
        for id2 in marker_ids[i+1:]:
            expected_mm = np.linalg.norm(template_3d[id1] - template_3d[id2])
            actual = np.linalg.norm(triangulated[id1] - triangulated[id2])
            if actual > 1e-6:
                scale = expected_mm / actual
                edge_scales.append(scale)
                print(f"  {id1}-{id2}: {actual:.4f} units -> {expected_mm:.1f}mm (scale={scale:.2f})")

    colmap_scale = np.median(edge_scales)
    print(f"\nMedian scale: {colmap_scale:.2f} mm per COLMAP unit")

    # Step 2: Get initial pose from PnP
    print("\nStep 2: Getting initial pose from PnP...")
    rvec_init, tvec_init, pnp_img = pnp_estimate_pose(all_markers, template_3d, images, cameras, colmap_scale)

    if rvec_init is None:
        print("ERROR: Could not estimate initial pose from PnP")
        # Fall back to triangulated centroid
        centroid = np.mean([triangulated[mid] for mid in marker_ids], axis=0)
        rvec_init = np.zeros(3)
        tvec_init = centroid
    else:
        print(f"  PnP from: {pnp_img}")
        print(f"  Initial rvec: {rvec_init}")
        print(f"  Initial tvec: {tvec_init}")

    # Step 3: Global optimization
    print("\nStep 3: Running global optimization...")

    # Initial parameters: [rx, ry, rz, tx, ty, tz, scale]
    x0 = np.concatenate([rvec_init, tvec_init, [colmap_scale]])

    result = minimize(
        compute_reprojection_error,
        x0,
        args=(all_markers, template_3d, images, cameras),
        method='Powell',
        options={'maxiter': 5000, 'disp': True}
    )

    print(f"\nOptimization result: {result.message}")
    print(f"Final mean reprojection error: {result.fun:.2f} pixels")

    # Extract optimized parameters
    rvec_opt = result.x[:3]
    tvec_opt = result.x[3:6]
    scale_opt = result.x[6]
    R_world = rodrigues_to_rotmat(rvec_opt)

    print(f"\nOptimized parameters:")
    print(f"  Scale: {scale_opt:.2f} mm per COLMAP unit")
    print(f"  Translation: ({tvec_opt[0]:.4f}, {tvec_opt[1]:.4f}, {tvec_opt[2]:.4f})")

    # Transform marker centers to world coordinates
    markers_world = {}
    print("\nMarker positions in COLMAP coordinates:")
    for mid in marker_ids:
        pt_board = template_3d[mid] / scale_opt
        pt_world = R_world @ pt_board + tvec_opt
        markers_world[mid] = pt_world
        print(f"  ID {mid}: ({pt_world[0]:.4f}, {pt_world[1]:.4f}, {pt_world[2]:.4f})")

    # Verify geometry
    print("\nVerifying geometry:")
    for i, id1 in enumerate(marker_ids):
        for id2 in marker_ids[i+1:]:
            expected_mm = np.linalg.norm(template_3d[id1] - template_3d[id2])
            actual = np.linalg.norm(markers_world[id1] - markers_world[id2])
            actual_mm = actual * scale_opt
            error = abs(actual_mm - expected_mm)
            print(f"  {id1} <-> {id2}: {actual_mm:.1f}mm (expected: {expected_mm:.1f}mm, error: {error:.1f}mm)")

    # Save output PLY
    print(f"\n{'='*60}")
    colors = {99: (255, 0, 0), 352: (0, 255, 0), 1676: (0, 0, 255), 2067: (255, 255, 0)}

    with open(args.output_ply, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex 4\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for mid in marker_ids:
            pos = markers_world[mid]
            c = colors.get(mid, (255, 255, 255))
            f.write(f"{pos[0]} {pos[1]} {pos[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"Saved: {args.output_ply}")

    # Regenerate combined point cloud
    print("\nRegenerating combined point cloud with markers...")
    recon_ply = os.path.join(os.path.dirname(args.output_ply), 'reconstruction.ply')
    combined_ply = os.path.join(os.path.dirname(args.output_ply), 'combined_with_markers.ply')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    combine_script = os.path.join(script_dir, 'combine_pointclouds.py')

    import subprocess
    subprocess.run([
        'python3', combine_script,
        '--reconstruction_ply', recon_ply,
        '--markers_ply', args.output_ply,
        '--output_ply', combined_ply,
        '--marker_radius', '0.03'
    ])

    # Save scale info
    scale_file = os.path.join(os.path.dirname(args.output_ply), 'scale_info.json')
    with open(scale_file, 'w') as f:
        json.dump({
            'scale_factor': float(scale_opt),
            'units': 'mm per COLMAP unit',
            'method': 'global_optimization',
            'final_reprojection_error_px': float(result.fun),
            'marker_centers_3d': {str(k): v.tolist() for k, v in markers_world.items()},
            'triangulated_positions': {str(k): v.tolist() for k, v in triangulated.items()}
        }, f, indent=2)
    print(f"Saved: {scale_file}")


if __name__ == '__main__':
    main()
