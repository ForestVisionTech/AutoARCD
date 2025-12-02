# iPhone + ChArUco Board Log Reconstruction Plan

## Your Setup

- **Board Type**: AprilTag 36h10 dictionary (NOT ChArUco - see note below)
- **Edge Point Distance**: 105mm between edge points
- **Camera**: iPhone
- **Goal**: 3D reconstruction of log cross-section with real-world scale

> **Important Note**: Your image shows **AprilTag markers**, not ChArUco. AprilTag is a different fiducial marker system. The detection code needs to use AprilTag detection, not ChArUco detection. OpenCV's `aruco` module can detect AprilTag dictionaries.

---

## Key Question: What Needs to Change?

### Answer: **Only the detection step needs to change**

Here's why:

```
CURRENT FLOW (Chessboard):
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. chessboard_extractor                                                    │
│    └─> Detects corners, writes to database as ChessboardCorners           │
│        • corners_2d: pixel positions                                      │
│        • corners_3d: template positions (0,0,0), (1,0,0), etc.           │
│                                                                            │
│ 2. feature_extractor                                                       │
│    └─> Reads ChessboardCorners from database                              │
│    └─> Writes them alongside SIFT keypoints                               │
│                                                                            │
│ 3. database_cache.cc (lines 162-184)                                      │
│    └─> Reads keypoints + corners                                          │
│    └─> Creates Shape2D from corners_2d                                    │
│    └─> Groups by corners_3d (same 3D template = same shape)               │
│    └─> Builds ShapeCorrespondenceGraph                                    │
│                                                                            │
│ 4. mapper (IncrementalTriangulator)                                        │
│    └─> Triangulates Shape3D from Shape2D observations                     │
│    └─> Estimates Sim3d (scale + rotation + translation)                   │
│                                                                            │
│ 5. align                                                                   │
│    └─> Uses Shape3D.sim.scale to get real-world scale                    │
└────────────────────────────────────────────────────────────────────────────┘
```

**The key insight**: Steps 3-5 don't care HOW the corners were detected. They only need:
1. `corners_2d`: 2D pixel positions of detected corners
2. `corners_3d`: 3D template positions (known geometry)

If you provide these in the same format, everything else works automatically.

---

## What You Need to Implement

### Step 1: AprilTag Corner Detection

Replace the chessboard detection with AprilTag detection that outputs the same format:

```
ChessboardCorners = pair<vector<Eigen::Vector2d>, vector<Eigen::Vector3d>>
                         ^                        ^
                         |                        |
                    2D pixels              3D template (known)
```

For your 2x2 AprilTag board (4 markers), each marker has 4 corners = 16 total corner points.

### AprilTag 36h10 Corner Layout

```
┌─────────────────────────────────────────┐
│  Marker 0          │   Marker 1         │
│  ┌───────┐         │   ┌───────┐        │
│  │ 0 1   │         │   │ 4 5   │        │
│  │ 3 2   │         │   │ 7 6   │        │
│  └───────┘         │   └───────┘        │
├────────────────────┼────────────────────┤
│  Marker 2          │   Marker 3         │
│  ┌───────┐         │   ┌───────┐        │
│  │ 8 9   │         │   │12 13  │        │
│  │11 10  │         │   │15 14  │        │
│  └───────┘         │   └───────┘        │
└─────────────────────────────────────────┘
```

### 3D Template Coordinates (in millimeters, scaled by 105mm edge distance)

If edge-to-edge is 105mm and you have 2x2 markers, the marker size determines the spacing.

For example, if each marker is 50mm and spacing is 5mm:
- Total width = 50 + 5 + 50 = 105mm (matches your 105mm)

Template coordinates (origin at top-left corner of marker 0):
```python
# Example - adjust based on your actual marker size
marker_size = 50.0  # mm
gap = 5.0           # mm between markers
scale = 105.0       # mm (your edge distance)

# Marker 0 corners (top-left marker)
corners_3d[0] = (0, 0, 0)
corners_3d[1] = (marker_size, 0, 0)
corners_3d[2] = (marker_size, marker_size, 0)
corners_3d[3] = (0, marker_size, 0)

# Marker 1 corners (top-right marker)
offset_x = marker_size + gap
corners_3d[4] = (offset_x, 0, 0)
# ... etc
```

---

## Implementation Options

### Option A: Python Script (Simplest, No C++ Changes)

Create a Python script that:
1. Detects AprilTag markers
2. Writes corners to a text file
3. Import using existing `chessboard_importer`

### Option B: C++ AprilTag Detector (Cleaner Integration)

Add a new detector alongside existing chessboard detector.

### Option C: Modify Existing Chessboard Detector (Not Recommended)

Would break existing functionality.

---

## Recommended: Option A - Python Script

### Why Python First?

1. **Faster iteration**: Debug detection without recompiling
2. **AprilTag libraries**: Python has excellent AprilTag support (`apriltag` or `pupil-apriltags`)
3. **Same database format**: Output matches what COLMAP expects
4. **No C++ changes needed**: Everything else works automatically

### Script Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IPHONE_APRILTAG_WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. ORGANIZE PHOTOS                                                     │
│     └─> Copy iPhone photos to project_path/images/                     │
│                                                                         │
│  2. DETECT APRILTAG CORNERS (Python)                                    │
│     └─> For each image:                                                │
│         - Detect AprilTag 36h10 markers                                │
│         - Extract 4 corners per marker                                 │
│         - Map to known 3D coordinates (105mm spacing)                  │
│         - Write to corners/{image_name}.txt                            │
│                                                                         │
│  3. CREATE DATABASE                                                     │
│     └─> colmap database_creator --database_path database.db            │
│                                                                         │
│  4. IMPORT CORNERS                                                      │
│     └─> colmap chessboard_importer                                     │
│         --database_path database.db                                    │
│         --import_path corners/                                         │
│                                                                         │
│  5. EXTRACT FEATURES                                                    │
│     └─> colmap feature_extractor (SIFT for the log itself)            │
│                                                                         │
│  6. MATCH FEATURES                                                      │
│     └─> colmap exhaustive_matcher (or sequential_matcher)              │
│                                                                         │
│  7. MAPPER                                                              │
│     └─> colmap mapper                                                  │
│         - Reconstructs camera poses                                    │
│         - Triangulates Shape3D from AprilTag corners                   │
│         - Shape3D.sim.scale gives real-world scale                    │
│                                                                         │
│  8. DENSE RECONSTRUCTION (Optional)                                     │
│     └─> colmap image_undistorter + patch_match_stereo                 │
│                                                                         │
│  9. APPLY SCALE                                                         │
│     └─> Scale reconstruction by 1/sim.scale                           │
│     └─> Export to PLY with real-world coordinates (mm)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Debug Plan

### Phase 1: Verify AprilTag Detection

1. Install `pupil-apriltags` or `apriltag` Python package
2. Write detection script
3. Visualize detected corners overlaid on image
4. Verify 3D template coordinates are correct

### Phase 2: Single Image Camera Calibration

1. Use AprilTag corners for camera intrinsic estimation
2. Verify focal length matches iPhone EXIF data
3. Verify distortion is reasonable

### Phase 3: Two-Image Reconstruction

1. Take 2 photos of log + AprilTag board
2. Run full pipeline
3. Verify Shape3D is triangulated
4. Check scale factor

### Phase 4: Full Reconstruction

1. Take 20-50 photos around log
2. Run full pipeline
3. Extract scaled point cloud
4. Measure known dimension to verify scale

---

## File Format for Corner Import

The existing `chessboard_importer` expects text files in this format:

```
# First line: num_rows num_cols
3 5
# Following lines: px py x y z
1234.5 567.8 0.0 0.0 0.0
1256.7 568.2 50.0 0.0 0.0
...
```

For AprilTag, you'd output:
```
# 4 markers x 4 corners = 16 points, arranged as 4 rows x 4 cols conceptually
4 4
# Pixel X, Pixel Y, 3D X (mm), 3D Y (mm), 3D Z (mm)
523.4 189.2 0.0 0.0 0.0
578.9 190.1 50.0 0.0 0.0
...
```

---

## Key Parameters for Your Setup

```python
# AprilTag 36h10 configuration
TAG_FAMILY = "tag36h10"

# Your physical board dimensions
EDGE_DISTANCE_MM = 105.0  # Distance between outermost corners

# Derived parameters (adjust based on your actual board)
# If you have 2x2 markers with edge distance = 105mm:
# - Each marker might be ~45mm
# - Gap between markers might be ~15mm
# - Adjust these to match your physical board

MARKER_SIZE_MM = 45.0  # Size of each marker (measure this!)
GAP_MM = 15.0          # Gap between markers (measure this!)

# Verification:
# 2 * MARKER_SIZE_MM + GAP_MM should equal EDGE_DISTANCE_MM
# 2 * 45 + 15 = 105 ✓
```

---

## What Doesn't Need to Change

1. **database.cc/h**: ChessboardCorners format works for any corner pattern
2. **database_cache.cc**: Creates Shape2D from any ChessboardCorners
3. **shape_triangulation.cc**: Triangulates any Shape2D observations
4. **incremental_triangulator.cc**: Handles shapes generically
5. **bundle_adjustment.cc**: Optimizes shapes generically
6. **align.cc**: Uses Shape3D.sim.scale (works with any shape)

---

## What You MIGHT Need to Change (Optional)

### If you want native C++ AprilTag support:

1. Add AprilTag library dependency (CMake)
2. Create `apriltag_extraction.cc` similar to `chessboard_extraction.cc`
3. Add command `colmap apriltag_extractor`

But this is **not required** - the Python approach works fine.

---

## Summary

| Component | Change Needed? | Why |
|-----------|---------------|-----|
| AprilTag detection | **YES** | New detector required |
| Database format | No | ChessboardCorners works |
| Feature extraction | No | SIFT still works for log |
| Shape triangulation | No | Generic algorithm |
| Bundle adjustment | No | Generic optimization |
| Scale extraction | No | Uses Shape3D.sim.scale |
| Alignment | Maybe* | Only if you have front/back setup |

*For single-camera iPhone setup, you don't need the align step. You directly use Shape3D.sim.scale.

---

## Next Steps

1. I'll create a Python detection script for AprilTag 36h10
2. I'll create a bash/Python workflow script
3. We'll test with your image first
4. Then iterate on the full reconstruction

Does this plan make sense? Should I proceed with creating the detection script?
