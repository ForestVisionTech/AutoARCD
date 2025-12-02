# ForestVision COLMAP Customization Analysis

## Overview

This document analyzes the highly customized version of COLMAP developed for ForestVision's log measurement system. The customization enables real-scale 3D reconstruction of logs using chessboard calibration patterns to establish ground truth scale.

---

## 1. Git History Analysis (May 2025 - December 2025)

### Key Commits Timeline

| Date | Commit | Description |
|------|--------|-------------|
| 2025-05-02 | `c10f0cce` | Initial chessboard calibration added |
| 2025-05-04 | `5ba8f664` | Added calibration functionality |
| 2025-05-19 | `32768fe6` | Control shape added |
| 2025-05-28 | `d3abbd1a` | Shape triangulation completed |
| 2025-06-02 | `c4c2df58` | Shape retriangulation and IO finalized |
| 2025-07-02 | `2d13da70` | Chessboard mesh integrated into workflow |
| 2025-07-09 | `d3d53dc9` | Chessboard detection tuning with `findChessboardCornersSB` |
| 2025-09-07 | `016926c4` | Still image logic added |
| 2025-09+ | Various | OpenMVS integration, VisMVSNet, instance mapping |

### Total Changes (7 months)
- **169 files changed**
- **+22,103 insertions, -561 deletions**
- Major new subsystems: Shape tracking, chessboard extraction, alignment, interpolation

---

## 2. Architecture: How the System Works

### 2.1 Core Data Structures

#### Shape2D (`src/colmap/scene/shape2d.h`)
```cpp
struct Shape2D {
  std::vector<Eigen::Vector2d> xy;  // Corner points in image coordinates
  shape3D_t shape3D_id;              // Link to 3D shape (like point3D_id)
};
```

#### Shape3D (`src/colmap/scene/shape3d.h`)
```cpp
struct Shape3D {
  Sim3d sim;                          // Scale + Rotation + Translation
  std::vector<Eigen::Vector3d> vertices;  // Template vertices (unit scale)
  double error;                       // Mean reprojection error
  ShapeTrack track;                   // Which images observe this shape
};
```

**Key Insight**: The `Sim3d` contains the **scale factor** that transforms unit-space vertices to real-world coordinates. This is how real scale is extracted.

### 2.2 Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WORKFLOW.PY                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. CHESSBOARD EXTRACTION                                               │
│     └─> colmap chessboard_extractor                                     │
│         - Detects chessboard corners in all images                      │
│         - Uses OpenCV findChessboardCorners / findChessboardCornersSB   │
│         - Stores corners in database                                    │
│                                                                         │
│  2. CAMERA CALIBRATION                                                  │
│     └─> colmap camera_calibrator                                        │
│         - Uses chessboard corners for intrinsic calibration             │
│         - Creates camera.db with calibrated intrinsics                  │
│                                                                         │
│  3. FEATURE EXTRACTION                                                  │
│     └─> colmap feature_extractor (or hloc)                              │
│         - Extracts SIFT/SuperPoint features                             │
│         - Also extracts "shapes" (chessboard corner patterns)           │
│         - ShapeExtraction.num_rows/num_cols parameters                  │
│                                                                         │
│  4. FEATURE MATCHING                                                    │
│     └─> Match features + merge shapes                                   │
│                                                                         │
│  5. INCREMENTAL MAPPING                                                 │
│     └─> colmap mapper                                                   │
│         - Standard COLMAP SfM pipeline                                  │
│         - PLUS: Shape triangulation in parallel                         │
│                                                                         │
│  6. ALIGNMENT (front + back rigs)                                       │
│     └─> colmap align                                                    │
│         - Aligns front and back camera reconstructions                  │
│         - Uses Shape3D (chessboard) as anchor points                    │
│         - Extracts real-world scale from chessboard                     │
│                                                                         │
│  7. POST-PROCESSING                                                     │
│     └─> Color extraction, mesh generation                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chessboard Detection Logic

### 3.1 Extraction (`src/colmap/calibration/chessboard.cc`)

```cpp
bool ChessboardExtractor::Extract(const Bitmap& bitmap,
                                  ChessboardCorners& corners, ...) {
  // 1. Convert to grayscale
  // 2. Optionally downscale for initial detection
  // 3. Use OpenCV findChessboardCorners with flags:
  //    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_ACCURACY
  // 4. Refine with cornerSubPix for sub-pixel accuracy
  // 5. Order corners consistently (top-left first)
  // 6. Create 3D template: corners3D[i] = (col_index, row_index, 0)
}
```

**Corner Ordering**: The `OrderCorners()` function ensures consistent orientation by checking the position of the first corner relative to horizontal/vertical neighbors.

### 3.2 Storage Format

Corners are stored as pairs:
- `ChessboardCorners2D`: `std::vector<Eigen::Vector2d>` - image pixel coordinates
- `ChessboardCorners3D`: `std::vector<Eigen::Vector3d>` - template coordinates (unit grid)

---

## 4. How Scale is Extracted and Applied

### 4.1 Shape Triangulation (`src/colmap/estimators/shape_triangulation.cc`)

The key is `EstimateShapeTriangulation`:

```cpp
bool EstimateShapeTriangulation(...) {
  // 1. For each pair of views observing the shape:
  //    - Triangulate 3D points from 2D observations
  //    - Collect source (template) and target (triangulated) point pairs

  // 2. Estimate Sim3d transformation:
  EstimateSim3d(source, target, sim3);
  // source = unit-space vertices (0,0,0), (1,0,0), (0,1,0), etc.
  // target = triangulated 3D points in reconstruction space

  // 3. The Sim3d contains:
  //    - scale: relates unit space to world space
  //    - rotation: orientation of shape in world
  //    - translation: position of shape origin in world
}
```

### 4.2 Scale Application (`src/colmap/exe/align.cc`)

```cpp
// Get the scale from the chessboard Shape3D
double scale1 = abs(1.0 / shape1.sim.scale);  // Inverse to go from recon to world
double scale2 = abs(1.0 / shape2.sim.scale);

// Apply scale to entire reconstruction
reconstruction1->Transform(
    Sim3d(scale1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()));
```

**The scale factor `sim.scale` represents "reconstruction units per chessboard unit"**

If a chessboard square is 30mm and `sim.scale = 0.033`, then:
- 1 chessboard unit = 30mm
- 1 reconstruction unit = 30mm * 0.033 = ~1mm

### 4.3 Bundle Adjustment with Shapes (`src/colmap/estimators/cost_functions.h`)

Shapes are jointly optimized during bundle adjustment:

```cpp
template <typename CameraModel>
class ReprojErrorShapeCostFunctor {
  // Parameters: cam_rotation, cam_translation, scale, shape_rotation, shape_translation, camera_params

  // Residual computation:
  // 1. Transform template vertex by shape Sim3d
  Eigen::Matrix<T, 3, 1> point3D =
      *scale * (EigenQuaternionMap<T>(rotation) * vertex_.cast<T>()) +
      EigenVector3Map<T>(translation);

  // 2. Project into camera
  // 3. Compute reprojection error
};
```

---

## 5. Image Analysis: IMG_9023.JPG

### What's in the Image

The image shows:

1. **Log Cookie (Cross-Section)**
   - A circular cross-section of a cut log
   - Shows tree rings, bark, and internal wood grain
   - Labeled with "016" identifier
   - Approximately 20-25cm diameter
   - Natural brown/orange wood color with darker bark edge

2. **ChArUco Board** (not "Chirico")
   - A 2x2 arrangement of ArUco markers
   - Black and white pattern on white background
   - Each marker is a unique binary pattern for identification
   - This is NOT a traditional chessboard - it's an ArUco marker board

3. **Background**
   - White/gray surface (likely a table)
   - Indoor setting with door visible in background

### ChArUco vs Chessboard

| Feature | Chessboard | ChArUco |
|---------|------------|---------|
| Pattern | Regular black/white squares | ArUco markers with unique IDs |
| Detection | OpenCV `findChessboardCorners` | OpenCV `cv2.aruco.detectMarkers` |
| Robustness | Requires full visibility | Partial visibility OK |
| Identification | No unique IDs | Each marker has unique ID |
| Scale | Known square size | Known marker + square size |

---

## 6. Adapting for ChArUco Board with iPhone

### 6.1 Key Differences from Current System

The current system uses:
- Standard chessboard with `findChessboardCorners`
- Fixed grid layout (e.g., 8x11 inner corners)
- All corners must be visible

ChArUco boards offer:
- ArUco markers with unique IDs embedded in chessboard
- Partial visibility detection
- More robust corner detection via marker localization

### 6.2 Recommended Implementation Approach

#### Option A: Modify Existing Chessboard Extractor

```cpp
// In chessboard.cc, add ChArUco detection path
bool ChArUcoExtractor::Extract(const Bitmap& bitmap, ChArUcoCorners& corners) {
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::CharucoBoard> board =
        cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, dictionary);

    // Detect ArUco markers first
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<int> markerIds;
    cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds);

    // Then interpolate ChArUco corners
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    cv::aruco::interpolateCornersCharuco(
        markerCorners, markerIds, gray, board, charucoCorners, charucoIds);

    // Convert to existing format with known 3D positions
    for (size_t i = 0; i < charucoIds.size(); ++i) {
        corners2D[charucoIds[i]] = Eigen::Vector2d(charucoCorners[i].x, charucoCorners[i].y);
        corners3D[charucoIds[i]] = board->getChessboardCorners()[charucoIds[i]];
    }
}
```

#### Option B: Python Pre-processing Pipeline

Create a Python script that runs before the COLMAP workflow:

```python
import cv2
import cv2.aruco as aruco
import numpy as np

def detect_charuco_corners(image_path, output_path):
    """
    Detect ChArUco corners and output in COLMAP-compatible format
    """
    # Define board parameters (adjust to match your physical board)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard_create(
        squaresX=5,      # Number of squares in X
        squaresY=7,      # Number of squares in Y
        squareLength=0.04,  # Square side length in meters
        markerLength=0.02,  # Marker side length in meters
        dictionary=dictionary
    )

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

    if ids is not None:
        # Interpolate CharUco corners
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        if ret > 4:  # Need minimum corners for pose estimation
            # Get 3D coordinates of detected corners
            obj_points = board.getChessboardCorners()[charuco_ids.flatten()]

            # Write to COLMAP format
            with open(output_path, 'w') as f:
                f.write(f"{len(charuco_ids)} 1\n")  # num_corners, dummy
                for i, (corner, obj_pt) in enumerate(zip(charuco_corners, obj_points)):
                    f.write(f"{corner[0][0]} {corner[0][1]} {obj_pt[0]} {obj_pt[1]} {obj_pt[2]}\n")

            return True
    return False
```

### 6.3 Integration with Existing Workflow

Modify `workflow.py`:

```python
def process_rig(args, rig_name):
    # ... existing code ...

    # NEW: Pre-process ChArUco detection
    if args.use_charuco:
        charuco_detection_command = \
            f"python detect_charuco.py " \
            f"--images_folder {images_folder} " \
            f"--output_folder {os.path.join(rig_folder, 'charuco_corners')} " \
            f"--squares_x {args.charuco_squares_x} " \
            f"--squares_y {args.charuco_squares_y} " \
            f"--square_length {args.charuco_square_length} " \
            f"--marker_length {args.charuco_marker_length} "
        os.system(charuco_detection_command)

        # Import detected corners as chessboard corners
        chessboard_command = \
            f"{args.colmap_exe_path} chessboard_importer " \
            f"--database_path {database_path} " \
            f"--image_path {images_folder} " \
            f"--import_path {os.path.join(rig_folder, 'charuco_corners')} "
    else:
        # Original chessboard detection
        chessboard_command = ...
```

### 6.4 iPhone Photo Considerations

For iPhone photos:

1. **EXIF Data**: iPhone stores focal length in EXIF - extract with:
   ```python
   from PIL import Image
   from PIL.ExifTags import TAGS
   img = Image.open('IMG_9023.JPG')
   exif = {TAGS.get(k, k): v for k, v in img._getexif().items()}
   focal_length_mm = exif['FocalLength']
   ```

2. **Camera Model**: Use `OPENCV` or `SIMPLE_RADIAL` camera model:
   ```
   --ImageReader.camera_model OPENCV
   --ImageReader.single_camera 1
   ```

3. **Image Quality**: iPhone ProRAW or HEIC offers higher quality; convert to JPEG/PNG for COLMAP

4. **Lens Distortion**: iPhone wide-angle has noticeable distortion - ensure proper calibration

---

## 7. Edge Detection for ChArUco Board

Since you mentioned "eye detection specifically for the ChArUco board", I interpret this as detecting the board edges/boundary:

### 7.1 Board Boundary Detection

```python
def detect_charuco_boundary(image, charuco_corners, charuco_ids, board):
    """
    Detect the outer boundary of the ChArUco board
    """
    if len(charuco_corners) < 4:
        return None

    # Get the expected corner positions on the board
    all_corners = board.getChessboardCorners()

    # Find detected corners at board edges
    squares_x, squares_y = board.getChessboardSize()
    edge_ids = []

    # Top edge (row 0)
    for x in range(squares_x - 1):
        edge_ids.append(x)
    # Bottom edge
    for x in range(squares_x - 1):
        edge_ids.append((squares_y - 2) * (squares_x - 1) + x)
    # Left edge
    for y in range(squares_y - 1):
        edge_ids.append(y * (squares_x - 1))
    # Right edge
    for y in range(squares_y - 1):
        edge_ids.append(y * (squares_x - 1) + (squares_x - 2))

    # Filter detected corners on edges
    edge_corners = []
    for i, cid in enumerate(charuco_ids.flatten()):
        if cid in edge_ids:
            edge_corners.append(charuco_corners[i][0])

    if len(edge_corners) >= 4:
        # Fit convex hull or rectangle
        hull = cv2.convexHull(np.array(edge_corners))
        return hull

    return None
```

### 7.2 Scale from Known Board Dimensions

```python
def compute_scale_from_charuco(charuco_corners, charuco_ids, board):
    """
    Compute pixels-per-meter scale from detected ChArUco corners
    """
    obj_points = board.getChessboardCorners()

    # Find adjacent corner pairs
    scales = []
    detected_set = set(charuco_ids.flatten())
    squares_x = board.getChessboardSize()[0] - 1

    for i, cid in enumerate(charuco_ids.flatten()):
        # Check horizontal neighbor
        if cid + 1 in detected_set and (cid + 1) % squares_x != 0:
            j = np.where(charuco_ids.flatten() == cid + 1)[0][0]
            pixel_dist = np.linalg.norm(charuco_corners[i] - charuco_corners[j])
            world_dist = np.linalg.norm(obj_points[cid] - obj_points[cid + 1])
            scales.append(pixel_dist / world_dist)

        # Check vertical neighbor
        if cid + squares_x in detected_set:
            j = np.where(charuco_ids.flatten() == cid + squares_x)[0][0]
            pixel_dist = np.linalg.norm(charuco_corners[i] - charuco_corners[j])
            world_dist = np.linalg.norm(obj_points[cid] - obj_points[cid + squares_x])
            scales.append(pixel_dist / world_dist)

    return np.median(scales) if scales else None
```

---

## 8. Summary: Steps to Implement ChArUco Support

1. **Create ChArUco detector script** (`scripts/python/detect_charuco.py`)
   - Detect ArUco markers
   - Interpolate ChArUco corners
   - Output in COLMAP-compatible format

2. **Modify workflow.py**
   - Add `--use_charuco` flag
   - Add board dimension parameters
   - Call ChArUco detection before feature extraction

3. **Option: Add native C++ support**
   - Create `charuco_extraction.cc` alongside `chessboard_extraction.cc`
   - Integrate with existing `ChessboardExtractorController`

4. **For iPhone photos**:
   - Extract EXIF focal length
   - Use appropriate camera model
   - Consider image preprocessing for better detection

5. **Scale calibration**:
   - Measure physical ChArUco board dimensions precisely
   - Pass to detector for accurate scale computation

---

## 9. File Reference

Key files for understanding and modification:

| File | Purpose |
|------|---------|
| `src/colmap/calibration/chessboard.cc` | Chessboard corner detection |
| `src/colmap/scene/shape2d.h`, `shape3d.h` | Shape data structures |
| `src/colmap/estimators/shape_triangulation.cc` | Shape 3D estimation |
| `src/colmap/estimators/cost_functions.h` | Bundle adjustment with shapes |
| `src/colmap/exe/align.cc` | Scale extraction and alignment |
| `src/colmap/sfm/incremental_triangulator.cc` | Shape triangulation during mapping |
| `scripts/python/workflow.py` | Main processing pipeline |

---

*Document generated: December 2025*
*Analyzed COLMAP version: ForestVision fork (maxwell_dev branch)*
