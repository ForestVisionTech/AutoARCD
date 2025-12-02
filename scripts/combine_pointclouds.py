#!/usr/bin/env python3
"""
Combine Reconstruction and Marker Point Clouds

Merges the main reconstruction PLY with marker centers, making the markers
more visible by creating larger spherical clusters of points around each marker.

Usage:
    python combine_pointclouds.py \
        --reconstruction_ply /path/to/reconstruction.ply \
        --markers_ply /path/to/marker_centers.ply \
        --output_ply /path/to/combined.ply \
        --marker_radius 0.05
"""

import argparse
import numpy as np
from pathlib import Path
import struct


def read_ply(filepath: str) -> tuple:
    """
    Read a PLY file (ASCII or binary) and return vertices and colors.

    Returns:
        tuple: (vertices_array, colors_array)
    """
    vertices = []
    colors = []

    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        vertex_count = 0
        has_color = False
        is_binary = False
        is_little_endian = True
        properties = []

        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)

            if line.startswith('format'):
                if 'binary_little_endian' in line:
                    is_binary = True
                    is_little_endian = True
                elif 'binary_big_endian' in line:
                    is_binary = True
                    is_little_endian = False

            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])

            if line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2] if len(parts) > 2 else ""
                properties.append((prop_type, prop_name))
                if prop_name in ('red', 'green', 'blue'):
                    has_color = True

            if line == 'end_header':
                break

        # Read vertices
        if is_binary:
            # Build format string based on properties
            endian = '<' if is_little_endian else '>'
            fmt = endian
            prop_sizes = {'float': 'f', 'double': 'd', 'uchar': 'B', 'char': 'b',
                          'int': 'i', 'uint': 'I', 'short': 'h', 'ushort': 'H'}

            for prop_type, prop_name in properties:
                if prop_type in prop_sizes:
                    fmt += prop_sizes[prop_type]
                else:
                    fmt += 'f'  # Default to float

            record_size = struct.calcsize(fmt)

            for _ in range(vertex_count):
                data = f.read(record_size)
                if len(data) < record_size:
                    break
                values = struct.unpack(fmt, data)

                x, y, z = values[0], values[1], values[2]
                vertices.append([x, y, z])

                if has_color and len(values) >= 6:
                    r, g, b = int(values[3]), int(values[4]), int(values[5])
                    colors.append([r, g, b])
                else:
                    colors.append([128, 128, 128])
        else:
            # ASCII format
            for _ in range(vertex_count):
                line = f.readline().decode('ascii')
                parts = line.split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])

                if has_color and len(parts) >= 6:
                    r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                    colors.append([r, g, b])
                else:
                    colors.append([128, 128, 128])

    return np.array(vertices), np.array(colors)


def create_marker_sphere(center: np.ndarray, color: list, radius: float = 0.05,
                         num_points: int = 100) -> tuple:
    """
    Create a sphere of points around a marker center for better visibility.

    Args:
        center: 3D position of marker center
        color: RGB color for the marker
        radius: Radius of the sphere in reconstruction units
        num_points: Number of points to create in the sphere

    Returns:
        tuple: (vertices, colors)
    """
    # Generate points on a sphere using Fibonacci lattice
    vertices = []
    colors = []

    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(num_points):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / num_points)

        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)

        vertices.append([x, y, z])
        colors.append(color)

    # Also add a dense core at the center
    for _ in range(50):
        offset = np.random.randn(3) * radius * 0.3
        vertices.append(center + offset)
        colors.append(color)

    return np.array(vertices), np.array(colors)


def write_ply(filepath: str, vertices: np.ndarray, colors: np.ndarray):
    """Write a PLY file with vertices and colors."""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for v, c in zip(vertices, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    parser = argparse.ArgumentParser(description="Combine reconstruction and marker point clouds")
    parser.add_argument("--reconstruction_ply", required=True,
                        help="Path to main reconstruction PLY file")
    parser.add_argument("--markers_ply", required=True,
                        help="Path to marker centers PLY file")
    parser.add_argument("--output_ply", required=True,
                        help="Output combined PLY file")
    parser.add_argument("--marker_radius", type=float, default=0.05,
                        help="Radius of marker spheres in reconstruction units (default: 0.05)")
    parser.add_argument("--marker_points", type=int, default=200,
                        help="Number of points per marker sphere (default: 200)")

    args = parser.parse_args()

    print("Combining Point Clouds")
    print("=" * 50)

    # Read reconstruction
    print(f"\nReading reconstruction: {args.reconstruction_ply}")
    recon_vertices, recon_colors = read_ply(args.reconstruction_ply)
    print(f"  Vertices: {len(recon_vertices)}")

    # Read marker centers
    print(f"\nReading marker centers: {args.markers_ply}")
    marker_vertices, marker_colors = read_ply(args.markers_ply)
    print(f"  Markers: {len(marker_vertices)}")

    # Create expanded marker spheres with bright colors
    print(f"\nCreating marker spheres (radius={args.marker_radius}, points={args.marker_points})")

    # Define bright colors for each marker (matching detect_markers.py)
    marker_sphere_colors = [
        [255, 0, 0],    # Red - marker 0 (top-left, ID 99)
        [0, 255, 0],    # Green - marker 1 (top-right, ID 352)
        [0, 0, 255],    # Blue - marker 2 (bottom-left, ID 1676)
        [255, 255, 0],  # Yellow - marker 3 (bottom-right, ID 2067)
    ]

    all_marker_vertices = []
    all_marker_colors = []

    for i, center in enumerate(marker_vertices):
        color = marker_sphere_colors[i % len(marker_sphere_colors)]
        sphere_verts, sphere_colors = create_marker_sphere(
            center, color, args.marker_radius, args.marker_points
        )
        all_marker_vertices.append(sphere_verts)
        all_marker_colors.append(sphere_colors)
        print(f"  Marker {i}: center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), color={color}")

    # Combine all vertices and colors
    if all_marker_vertices:
        marker_verts_combined = np.vstack(all_marker_vertices)
        marker_colors_combined = np.vstack(all_marker_colors)
    else:
        marker_verts_combined = np.array([]).reshape(0, 3)
        marker_colors_combined = np.array([]).reshape(0, 3)

    combined_vertices = np.vstack([recon_vertices, marker_verts_combined])
    combined_colors = np.vstack([recon_colors, marker_colors_combined])

    print(f"\nCombined point cloud:")
    print(f"  Reconstruction points: {len(recon_vertices)}")
    print(f"  Marker points: {len(marker_verts_combined)}")
    print(f"  Total points: {len(combined_vertices)}")

    # Write output
    print(f"\nWriting combined PLY: {args.output_ply}")
    write_ply(args.output_ply, combined_vertices, combined_colors)

    print("\nDone!")


if __name__ == "__main__":
    main()
