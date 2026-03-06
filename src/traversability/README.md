# Traversability Documentation

### 1. `tilt_compensate.py`
#### What it does

This module corrects a 3D point cloud for the robot's **pitch and roll** tilt, so that the points are expressed in a gravity-aligned frame. This is a preprocessing step in the offline traversability pipeline.

Odometry gives the full 6-DOF orientation (pitch, roll, yaw) of the robot. For traversability analysis you typically want the terrain to appear "flat" even when the robot is on a slope, so pitch and roll need to be factored out. Yaw (heading) is left untouched because it has no effect on perceived slope.

#### How it works

1. **Normalize the quaternion** — the incoming `[qx, qy, qz, qw]` orientation from odometry is normalized to unit length.

2. **Extract yaw** — the yaw angle (rotation about the world Z axis) is computed analytically from the quaternion using the standard `atan2` formula.

3. **Isolate pitch + roll** — a pure-yaw rotation `Rz(yaw)` is constructed and removed from the full orientation:

   ```
   q_pitch_roll = q_world_cam * Rz(yaw)^-1
   ```

   The result `q_pitch_roll` contains only the pitch and roll components.

4. **Rotate the point cloud** — the XYZ columns of every point are rotated by `q_pitch_roll`. Extra columns (e.g. intensity, ring index) are copied unchanged.


---

### 2. `voxel_filter.py`

#### What it does

Downsamples a 3D point cloud by dividing space into a regular grid of voxels and replacing all points that fall into the same voxel with a single representative point at the **voxel center**. Voxels with fewer than a minimum number of points are discarded as noise.

#### How it works

1. **Assign voxel indices** — each point's XYZ is divided by the configured voxel size and floor-divided to get an integer `(ix, iy, iz)` index, matching C++ voxel indexing for both positive and negative coordinates.

2. **Count occupancy** — `np.unique` groups points by their voxel index and counts how many points fall in each voxel.

3. **Filter sparse voxels** — voxels with fewer than `min_points_per_voxel` points are dropped. This removes isolated noise points.

4. **Snap to voxel center** — surviving points are replaced by their voxel center:

   ```
   center = (floor(xyz / voxel_size) + 0.5) * voxel_size
   ```

   Note: one output point is emitted **per input point** that passed the filter (not one per voxel), so the caller should deduplicate if needed.

#### Config keys

| Key | Default | Description |
|-----|---------|-------------|
| `voxel_size_x` | `0.05` | Voxel width along X (metres) |
| `voxel_size_y` | `0.05` | Voxel width along Y (metres) |
| `voxel_size_z` | `0.05` | Voxel height along Z (metres) |
| `min_points_per_voxel` | `3` | Minimum occupancy to keep a voxel. *After the point cloud is binned into voxels, any voxel that contains fewer than 3 points is thrown away entirely.* |



---

### 3. `polarize.py` 

#### What it does

Converts a Cartesian `[x, y, z]` point cloud into cylindrical polar coordinates `[r, theta, z]`, while filtering out points that are too close to the sensor or too high/low in Z. This representation is useful for range-based terrain analysis.

#### How it works

1. **Z filtering** — points with `|z| >= z_threshold` are discarded. This removes ceiling hits, high obstacles, and ground clutter far below the sensor plane.

2. **Compute r and theta** — for surviving points:

   ```
   r     = sqrt(x² + y²)   # horizontal distance from sensor
   theta = atan2(y, x)      # bearing angle in radians
   ```

3. **Min-range filtering** — points with `r <= min_range` are discarded. This removes returns from the robot's own body or sensor blind spot.

4. **Return** — surviving points are stacked as `[r, theta, z]` columns.

#### Config keys

| Key | Default | Description |
|-----|---------|-------------|
| `z_threshold` | `1.5` | Max absolute Z height to keep (metres) |
| `min_range`   | `0.1` | Min horizontal range to keep (metres) |

---

### 4. `traversability.py` 

#### What it does

It takes the polar `[r, theta, z]` point cloud and produces a **2D polar danger grid**:  a map where each cell contains a scalar danger score indicating how difficult that terrain patch is to traverse. Cells above a threshold are flagged as non-traversable.

#### How it works

**Step 1 — Build a polar height map**

Space is discretized into a 2D polar grid with configurable radial (`polar_grid_size_r`) and angular (`polar_grid_size_theta_deg`) bin sizes. For each `(r, theta)` bin, the **maximum Z** of all points falling in that bin is recorded. This gives a height map `H[r, theta]` in polar coordinates. Bins with no points are marked invalid and filled with 0 for the filter stage.
Todo: document how choice of polar_grid_size_r and polar_grid_size_theta_deg affects the computation.

Todo: document choice of  `height_map[~valid_mask] = 0.0`
Todo: document choice of `polar_grid_size_theta_deg` and `polar_grid_size_r` and its impact.
Todo: currently harcoding   

``` yaml
r_min_m: 0.3
r_max_m: 2.0
theta_min_deg: -45.0
theta_max_deg: 45.0
```
**Step 2 — Compute three terrain features**

Three features are derived from the height map `H`:

- **Slope** — the gradient magnitude of H in metric units:

  ```
  slope = arctan(sqrt((dH/dr)² + (dH/dθ)²))
  ```

  Cells where `slope > scrit_deg` are set to `inf` (immediately dangerous).

- **Roughness** — standard deviation of H over a 3×3 neighborhood. Cells where `roughness > rcrit_m` are set to `inf`.

- **Step height** — the maximum absolute height difference between a cell and any neighbor in a 5×5 window. A weighting factor scales it down by the fraction of neighbors that exceed `hcrit_m`, so isolated spikes are penalized less than wide steps. Cells where the result exceeds `hcrit_m` are set to `inf`.

**Step 3 — Combine into a danger score**

The three features are normalized by their critical thresholds and blended with fixed weights:

```
danger = 0.3 * (slope / scrit)  +  0.3 * (roughness / rcrit)  +  0.4 * (step_height / hcrit)
```

The `0.4` weight on step height reflects that abrupt discontinuities are the most important obstacle signal.

**Step 4 — Ray cast mask**

Depth points only mark bins where returns landed, leaving NaNs between the
sensor and the furthest return in a column. This helper applies occupancy-grid
style ray casting: for each theta bin, all radial bins closer than the furthest
non-traversable cell return are marked observed.

**Step 5 — Mask and threshold**

- Grid cells that had no input points are set to `NaN` (unknown, not dangerous) `valid_mask = ~np.isnan(height_map)`. 
- Cells with `danger > danger_threshold` are flagged as **non-traversable**.

#### Config keys

| Key | Default | Description |
|-----|---------|-------------|
| `polar_grid_size_r_m` | `0.10` | Radial bin size (metres) |
| `polar_grid_size_theta_deg` | `1.0` | Angular bin size (degrees) |
| `r_min_m` / `r_max_m` | data range | Radial extent of the grid |
| `theta_min_deg` / `theta_max_deg` | data range | Angular extent of the grid |
| `scrit_deg` | `30.0` | Critical slope angle (degrees) |
| `rcrit_m` | `0.10` | Critical roughness (metres) |
| `hcrit_m` | `0.20` | Critical step height (metres) |
| `danger_threshold` | `0.30` | Score above which a cell is non-traversable |

#### Output

```python
trav_grid, r_edges, theta_edges, height_map = compute_traversability(points, config)
```

| Return value     | Shape / Type | Description |
|------------------|--------------|-------------|
| `trav_grid`      | `(R, T)` float32 | Traversability map: `NaN` unknown, `0.0` traversable, `1.0` non-traversable |
| `r_edges`        | `(R+1,)` float32 | Radial bin edges |
| `theta_edges`    | `(T+1,)` float32 | Angular bin edges (radians) |
| `height_map`     | `(R, T)` float32 | Max-height polar map used for visualization/filtering |
