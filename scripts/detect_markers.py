#!/usr/bin/env python3
"""
AprilTag Marker Center Detection

Detects AprilTag 36h10 markers and saves their center positions with IDs.

Output format (JSON):
{
    "marker_id": [x_2d, y_2d],
    ...
}

Output format (TXT):
# marker_id x_2d y_2d x_3d y_3d z_3d
99 2500.5 3200.3 105.0 105.0 0.0
352 3100.2 3180.5 0.0 105.0 0.0
...

Board configuration:
    - AprilTag family: 36h10
    - Marker IDs: 99, 352, 1676, 2067
    - Center-to-center distance: 105mm
    - Layout: 2x2 grid

Usage:
    python detect_markers.py --image_path /path/to/images --output_path /path/to/output
"""

import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path

# Initialize OpenCV ArUco detector for AprilTag 36h10
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    print("Using OpenCV ArUco backend (DICT_APRILTAG_36h10)")
except AttributeError:
    print("ERROR: OpenCV ArUco with AprilTag support not available.")
    print("Install with: pip install opencv-contrib-python")
    exit(1)


class MarkerBoard:
    """
    Defines the AprilTag board layout and 3D positions.

    Board layout (looking at the board):

        +Y
        ^
        |   [99]----[352]
        |     |       |      105mm between centers
        |  [1676]--[2067]
        |
        +-------> +X

    Origin at marker 2067 (bottom-right when looking at board)
    """

    def __init__(self, marker_ids: list, center_distance_mm: float = 105.0):
        """
        Args:
            marker_ids: List of 4 marker IDs in order [top-left, top-right, bottom-left, bottom-right]
            center_distance_mm: Distance between adjacent marker centers
        """
        self.marker_ids = marker_ids
        self.center_distance = center_distance_mm

        # Define 3D positions (mm) for each marker
        # Origin at bottom-right marker, +X right, +Y up, +Z out of board
        self.positions_3d = {
            marker_ids[0]: np.array([0.0, center_distance_mm, 0.0]),           # top-left
            marker_ids[1]: np.array([center_distance_mm, center_distance_mm, 0.0]),  # top-right
            marker_ids[2]: np.array([0.0, 0.0, 0.0]),                          # bottom-left
            marker_ids[3]: np.array([center_distance_mm, 0.0, 0.0]),           # bottom-right
        }

    def get_3d_position(self, marker_id: int) -> np.ndarray:
        """Get the 3D position for a marker ID."""
        return self.positions_3d.get(marker_id)


def detect_markers(image: np.ndarray, expected_ids: list = None):
    """
    Detect AprilTag markers in an image.

    Args:
        image: BGR or grayscale image
        expected_ids: Optional list of expected marker IDs to filter

    Returns:
        dict: {marker_id: center_2d} for each detected marker
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect markers
    corners_list, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return {}

    # Extract centers
    detections = {}
    for i, marker_id in enumerate(ids.flatten()):
        if expected_ids is not None and marker_id not in expected_ids:
            continue

        corners = corners_list[i][0]  # Shape: (4, 2)
        center = np.mean(corners, axis=0)
        detections[int(marker_id)] = center

    return detections


def write_markers_txt(output_path: str, detections: dict, board: MarkerBoard):
    """
    Write marker centers to text file.

    Format:
        # marker_id x_2d y_2d x_3d y_3d z_3d
        99 2500.5 3200.3 0.0 105.0 0.0
        ...
    """
    with open(output_path, 'w') as f:
        f.write("# marker_id x_2d y_2d x_3d y_3d z_3d\n")
        for marker_id, center_2d in sorted(detections.items()):
            pos_3d = board.get_3d_position(marker_id)
            if pos_3d is not None:
                f.write(f"{marker_id} {center_2d[0]:.6f} {center_2d[1]:.6f} "
                       f"{pos_3d[0]:.6f} {pos_3d[1]:.6f} {pos_3d[2]:.6f}\n")


def write_markers_json(output_path: str, detections: dict, board: MarkerBoard):
    """
    Write marker centers to JSON file.

    Format:
    {
        "markers": {
            "99": {"center_2d": [x, y], "center_3d": [x, y, z]},
            ...
        },
        "board": {
            "center_distance_mm": 105.0,
            "marker_ids": [99, 352, 1676, 2067]
        }
    }
    """
    data = {
        "markers": {},
        "board": {
            "center_distance_mm": board.center_distance,
            "marker_ids": board.marker_ids
        }
    }

    for marker_id, center_2d in detections.items():
        pos_3d = board.get_3d_position(marker_id)
        if pos_3d is not None:
            data["markers"][str(marker_id)] = {
                "center_2d": center_2d.tolist(),
                "center_3d": pos_3d.tolist()
            }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def visualize_detections(image: np.ndarray, detections: dict, board: MarkerBoard) -> np.ndarray:
    """Create visualization of detected markers."""
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Colors for each marker (consistent by ID)
    colors = {
        board.marker_ids[0]: (0, 0, 255),    # Red - top-left
        board.marker_ids[1]: (0, 255, 0),    # Green - top-right
        board.marker_ids[2]: (255, 0, 0),    # Blue - bottom-left
        board.marker_ids[3]: (0, 255, 255),  # Yellow - bottom-right
    }

    for marker_id, center in detections.items():
        center_int = tuple(center.astype(int))
        color = colors.get(marker_id, (255, 255, 255))

        # Draw center point
        cv2.circle(vis, center_int, 15, color, -1)
        cv2.circle(vis, center_int, 20, (255, 255, 255), 3)

        # Draw marker ID
        cv2.putText(vis, f"ID:{marker_id}", (center_int[0] + 25, center_int[1] + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
        cv2.putText(vis, f"ID:{marker_id}", (center_int[0] + 25, center_int[1] + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw 3D position
        pos_3d = board.get_3d_position(marker_id)
        if pos_3d is not None:
            cv2.putText(vis, f"({pos_3d[0]:.0f},{pos_3d[1]:.0f})",
                       (center_int[0] + 25, center_int[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add summary
    cv2.putText(vis, f"Detected: {len(detections)} markers", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return vis


def process_image(image_path: str, output_dir: str, board: MarkerBoard,
                  visualize: bool = False, output_format: str = "txt") -> dict:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Could not read {image_path}"}

    # Detect markers
    detections = detect_markers(image, expected_ids=board.marker_ids)

    if len(detections) == 0:
        print(f"  {os.path.basename(image_path)}: NO markers detected")
        return {"success": False, "detected": 0}

    # Generate output filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Write output
    if output_format == "json":
        output_file = os.path.join(output_dir, f"{image_name}.json")
        write_markers_json(output_file, detections, board)
    else:
        output_file = os.path.join(output_dir, f"{image_name}.txt")
        write_markers_txt(output_file, detections, board)

    print(f"  {os.path.basename(image_path)}: {len(detections)} markers -> {os.path.basename(output_file)}")

    # Save visualization
    if visualize:
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis = visualize_detections(image, detections, board)
        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
        cv2.imwrite(vis_path, vis)

    return {
        "success": True,
        "detected": len(detections),
        "marker_ids": list(detections.keys()),
        "output_file": output_file
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect AprilTag marker centers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a directory of images
    python detect_markers.py --image_path ./images --output_path ./markers

    # With visualization
    python detect_markers.py --image_path ./images --output_path ./markers --visualize

    # JSON output format
    python detect_markers.py --image_path ./images --output_path ./markers --format json
        """
    )
    parser.add_argument("--image_path", required=True,
                        help="Path to image or directory of images")
    parser.add_argument("--output_path", required=True,
                        help="Output directory for marker files")
    parser.add_argument("--center_distance", type=float, default=105.0,
                        help="Center-to-center distance in mm (default: 105)")
    parser.add_argument("--marker_ids", type=str, default="99,352,1676,2067",
                        help="Marker IDs as top-left,top-right,bottom-left,bottom-right")
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                        help="Output format (default: txt)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")

    args = parser.parse_args()

    # Parse marker IDs
    marker_ids = [int(x.strip()) for x in args.marker_ids.split(",")]
    if len(marker_ids) != 4:
        print("ERROR: Exactly 4 marker IDs required (top-left, top-right, bottom-left, bottom-right)")
        exit(1)

    # Create board definition
    board = MarkerBoard(marker_ids, args.center_distance)

    print(f"\nAprilTag Marker Detector")
    print(f"=" * 50)
    print(f"Center distance: {args.center_distance}mm")
    print(f"Marker IDs: {marker_ids}")
    print(f"  Top-left:     {marker_ids[0]} -> (0, {args.center_distance})")
    print(f"  Top-right:    {marker_ids[1]} -> ({args.center_distance}, {args.center_distance})")
    print(f"  Bottom-left:  {marker_ids[2]} -> (0, 0)")
    print(f"  Bottom-right: {marker_ids[3]} -> ({args.center_distance}, 0)")
    print(f"Output format: {args.format}")
    print()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Process images
    if os.path.isfile(args.image_path):
        # Single image
        result = process_image(args.image_path, args.output_path, board,
                              args.visualize, args.format)
        if result["success"]:
            print(f"\nOutput: {result['output_file']}")
    else:
        # Directory
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(args.image_path)
                      if os.path.splitext(f.lower())[1] in extensions]

        print(f"Found {len(image_files)} images\n")

        success = 0
        for image_file in sorted(image_files):
            image_path = os.path.join(args.image_path, image_file)
            result = process_image(image_path, args.output_path, board,
                                  args.visualize, args.format)
            if result["success"]:
                success += 1

        print(f"\n{'='*50}")
        print(f"Processed: {success}/{len(image_files)} images")
        print(f"Output: {args.output_path}")

    # Write board definition file
    board_file = os.path.join(args.output_path, "board_definition.json")
    with open(board_file, 'w') as f:
        json.dump({
            "family": "AprilTag 36h10",
            "center_distance_mm": args.center_distance,
            "marker_positions": {
                str(k): v.tolist() for k, v in board.positions_3d.items()
            },
            "layout": {
                "top_left": marker_ids[0],
                "top_right": marker_ids[1],
                "bottom_left": marker_ids[2],
                "bottom_right": marker_ids[3]
            }
        }, f, indent=2)
    print(f"Board definition: {board_file}")


if __name__ == "__main__":
    main()
