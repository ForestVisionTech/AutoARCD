# Marker Localization Analysis Report

**Date:** December 2, 2024
**Dataset:** Log012 (80 images)
**Board:** AprilTag 36h10, markers 99, 352, 1676, 2067 with 105mm center-to-center spacing

---

## Executive Summary

The ARCD pipeline has two distinct tasks:

| Task | Status | Key Metric |
|------|--------|------------|
| 1. COLMAP Reconstruction | ✅ Working Well | 1.13 px reprojection error |
| 2. Marker Localization | ❌ Working Poorly | 899 px reprojection error |

**Root Cause:** ArUco marker center detections are geometrically inconsistent across views, despite COLMAP camera poses being accurate.

---

## Task 1: COLMAP Sparse Reconstruction

### What It Does
- Extracts SIFT features from all images
- Matches features across image pairs
- Runs Structure-from-Motion (SfM) to estimate camera poses and 3D points
- Bundle adjustment refines the solution

### Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Images registered | 80 / 80 | ✅ All images |
| 3D points | 55,940 | ✅ Dense coverage |
| Mean reprojection error | 1.13 px | ✅ Excellent |

### Verification Method

To verify camera pose accuracy, we tested whether COLMAP's own 3D points project correctly:

```python
# For each 3D point tracked across multiple views:
# 1. Get the 3D position from COLMAP
# 2. Project it into each observing camera
# 3. Compare to the 2D feature detection

# Results:
# - Reprojection errors: 0.5 - 1.3 pixels
# - Ray intersection errors: 0.0001 - 0.004 COLMAP units
```

**Conclusion:** Camera poses are accurate. The reconstruction is reliable.

---

## Task 2: Marker Center Localization

### What It Does
1. Detect AprilTag markers in each image using OpenCV ArUco detector
2. Compute marker centers from detected corners
3. Triangulate 3D positions using COLMAP camera poses
4. Or: Optimize board pose to minimize reprojection error

### Results

| Method | Reprojection Error | Assessment |
|--------|-------------------|------------|
| Direct triangulation | N/A (rays don't intersect) | ❌ Failed |
| Global optimization | 899 px | ❌ Very poor |
| Per-view PnP | 1-8 px per view | ⚠️ Good locally, inconsistent globally |

### Verification Method 1: Ray Intersection Analysis

For a 3D point to be accurately triangulated, viewing rays from different cameras must intersect at a single point.

```python
# For each marker, compute rays from all observing cameras
# Find the point closest to all rays
# Measure how far each ray passes from this point

# Results for COLMAP SIFT features:
# - Ray miss distance: 0.0001 - 0.004 units (excellent)

# Results for ArUco marker centers:
# - Ray miss distance: 0.3 - 1.1 units (100-300x worse)
```

**Interpretation:** Marker rays don't converge. The 2D detections are inconsistent across views.

### Verification Method 2: Per-View PnP Analysis

If detections were consistent, solving PnP for each view should give the same 3D board position.

```python
# For each image with 4 detected markers:
# 1. Solve PnP to find board pose in camera frame
# 2. Transform to world coordinates using COLMAP camera pose
# 3. Compare marker world positions across views

# Results:
# - Per-view reprojection: 1-8 pixels (good fit locally)
# - World position variation: >1000mm range (views disagree)
```

**Interpretation:** Each view's markers form a valid 105mm square, but they place that square in different 3D locations.

### Verification Method 3: Scale Consistency Check

Even if absolute positions are wrong, can we trust the scale?

```python
# Using optimized marker positions:
# - Edge distances: exactly 105mm (by construction - constrained)
# - Diagonal distances: exactly 148.49mm (by construction)

# Using per-view PnP scales:
# - Scale range: 131 - 208 mm/unit
# - Mean: 166.9 mm/unit
# - Std dev: 26.9 mm/unit (~16% uncertainty)

# Using triangulated positions (unconstrained):
# - Distances are NOT 105mm - geometry is distorted
```

**Interpretation:** Scale has ~15-20% uncertainty. The 105mm constraint is satisfied only because we enforce it.

---

## Detailed Evidence

### COLMAP Statistics (from reconstruction)

```
Sparse 3D point cloud:  55940 points
Registered images:      80 images
Mean reprojection:      1.13485 pixels
Mean track length:      4.44 observations per point
```

### Marker Detection Statistics

```
Images with markers detected: 43 / 80
Markers per image: 4 (when detected)
Detection method: cv2.aruco.detectMarkers (DICT_APRILTAG_36h10)
```

### Ray Intersection Comparison

| Point Type | Min Error | Max Error | Interpretation |
|------------|-----------|-----------|----------------|
| COLMAP SIFT features | 0.0001 units | 0.004 units | Rays converge precisely |
| ArUco marker centers | 0.3 units | 1.1 units | Rays don't converge |

This is a **100-300x difference** in geometric consistency.

### Per-View Scale Variation

```
View 1:  131 mm/unit
View 2:  145 mm/unit
View 3:  168 mm/unit
View 4:  189 mm/unit
View 5:  208 mm/unit
...
Mean:    166.9 mm/unit
Std:     26.9 mm/unit
```

---

## Tools and Code Used

### Detection Script
`arcd/scripts/detect_markers.py`
- Uses OpenCV ArUco detector
- Outputs marker corners and centers per image

### Localization Script
`arcd/scripts/localize_board_final.py`
- Reads COLMAP camera poses from `images.txt` and `cameras.txt`
- Reads marker detections from JSON files
- Attempts triangulation and global optimization
- Outputs `scale_info.json` and `marker_centers.ply`

### Verification Code (ad-hoc analysis)
```python
# Ray intersection analysis
def compute_ray_from_camera(camera, image_pose, pixel_2d):
    # Unproject 2D point to 3D ray in world coordinates
    ...

def ray_closest_point(rays):
    # Find point minimizing sum of squared distances to all rays
    ...

# Reprojection test
def project_point(point_3d, camera, image_pose):
    # Project 3D point to 2D using camera intrinsics and pose
    ...
```

---

## Judgment Criteria

### How We Know COLMAP Is Working

1. **Low reprojection error (1.13 px)**: Industry standard is <1.5 px for good reconstruction
2. **High registration rate (80/80)**: All images successfully placed
3. **Consistent ray geometry**: COLMAP's 3D points project back accurately

### How We Know Marker Localization Is Failing

1. **High reprojection error (899 px)**: 800x worse than COLMAP
2. **Rays don't intersect**: 100-300x worse geometric consistency
3. **Views disagree on position**: >1000mm variation in world coordinates
4. **Scale uncertainty**: ±15-20% when it should be <1%

---

## Possible Causes

1. **Marker too small in frame**: Less precise corner detection
2. **Lens distortion effects**: Marker appearance changes with viewing angle
3. **ArUco detector limitations**: Not designed for sub-pixel accuracy
4. **Marker planarity assumption**: Board may not be perfectly flat

---

## Potential Solutions

1. **Improve detection accuracy**
   - Use corner refinement (cv2.cornerSubPix)
   - Template matching for marker centers
   - Larger markers or closer camera positions

2. **Use textured calibration object**
   - Object with SIFT-trackable features
   - Known dimensions for scale reference
   - Let COLMAP track it naturally

3. **Hybrid approach**
   - Use ArUco for coarse detection
   - Refine with feature matching around marker corners

4. **Accept limitations**
   - Use COLMAP reconstruction as-is (it's accurate)
   - Apply approximate scale from marker spacing
   - Accept ~15-20% scale uncertainty

---

## Conclusion

The COLMAP sparse reconstruction is working correctly with excellent accuracy (1.13 px error). The problem lies specifically in ArUco marker center detection - the 2D positions are not geometrically consistent across different viewing angles, causing triangulation to fail despite accurate camera poses.

This is fundamentally a **marker detection quality issue**, not a camera calibration or reconstruction issue.
