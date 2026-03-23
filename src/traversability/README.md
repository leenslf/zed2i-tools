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

Input points `[r, θ, z]` are binned into a 2D polar grid. Each cell stores the **maximum z** (height) of all points falling in that `(r, θ)` bin:

$$H[i,j] = \begin{cases} \max\{z : (r, \theta, z) \in \text{bin}(i,j)\} & \text{if bin}(i,j) \text{ contains at least one point} \\ z_\text{fill} & \text{otherwise} \end{cases}$$

where $z_\text{fill} = -0.3\,\text{m}$.
Space is discretized into a 2D polar grid with configurable radial (`polar_grid_size_r`) and angular (`polar_grid_size_theta_deg`) bin sizes. For each `(r, theta)` bin, the **maximum Z** of all points falling in that bin is recorded. This gives a height map `H[r, theta]` in polar coordinates.


**i) Choice of `polar_grid_size_theta_deg` and `polar_grid_size_r`**

`polar_grid_size_r_m: 0.10` seems fine. 
`polar_grid_size_theta_deg` however is a bit of a pickle, I can't seem to set on a value that performs well always without leaving holes in the traversability grid. Too small, the grid looks very sparse and begins to compute traversability incorrectly, too big and the grid cells appear too coarse. 

**ii) *To-do: currently harcoding***

``` yaml
r_min_m: 0.3
r_max_m: 2.0
theta_min_deg: -45.0
theta_max_deg: 45.0
``` 
Perhaps these ranges should be detected dynamically, the restriction is currently made for simplifying interpretation. 


**iii) Fill empty NaN values with a negative number**



The height values are interpreted in the tilt-compensated camera frame:

- `z > 0`: point is above the camera
- `z < 0`: point is below the camera
- `z = 0`: point lies in the horizontal plane through the camera center

Bins with no points are marked invalid. In the current Python implementation they are temporarily filled with `-0.3` before the local terrain filters are applied. The value is chosen to represent terrain somewhat below the camera plane, which is consistent with a camera mounted above the robot's contact surface. 

After danger estimation, those originally invalid bins are masked back to `NaN`, but nearby valid bins may still reflect the influence of this fill value because the filters operate on local neighborhoods.

**Step 2 — Compute three terrain features**

Three features are derived from the height map $H$. Any feature exceeding its critical threshold is set to $\infty$, making the cell unconditionally non-traversable.

**Slope** — gradient magnitude in metric units, with the angular component converted from z/rad to z/m via arc-length:

$$\sigma[i,j] = \arctan\!\sqrt{\left(\frac{\partial H}{\partial r}\right)^2 + \left(\frac{1}{\bar{r}_i}\frac{\partial H}{\partial \theta}\right)^2}$$

where $\bar{r}_i = r_{\text{min}} + (i+\tfrac{1}{2})\Delta r$ is the radial bin centre. 

$\sigma$ is clipped to $\infty$ when $\sigma > \sigma_\text{crit}$.

**Roughness** — standard deviation of $H$ over the $3\times3$ neighbourhood $\mathcal{N}_3$:

$$\rho[i,j] = \text{std}\!\left\lbrace H[i+p,\,j+q] : (p,q)\in\mathcal{N}_3\right\rbrace$$

$\rho$ is clipped to $\infty$ when $\rho > \rho_\text{crit}$.

**Step height** — for each cell, every neighbour $k$ in the $5\times5$ window $\mathcal{N}_5$ is tested. Neighbour $k$ qualifies as a step if:

$$\delta z_k = |H[i,j] - H_k| > h_\text{crit} \quad\text{and}\quad \arctan2(\delta z_k,\, d^\text{xy}_k) > \sigma_\text{crit}$$

where $d^\text{xy}_k = \lVert(X[i,j]-X_k,\; Y[i,j]-Y_k)\rVert$ is the true Cartesian horizontal distance, with $X = r\cos\theta$, $Y = r\sin\theta$.

**Why Cartesian distance?** The pairwise slope $\arctan2(\delta z_k, d^\text{xy}_k)$ is a physical angle and requires $d^\text{xy}_k$ to be a real metric distance in metres. Polar grid bins are not isometric: radial neighbours are always $\Delta r$ metres apart, but angular neighbours are $\bar{r}_i \cdot \Delta\theta$ metres apart, which grows with range. Using grid-index distance would therefore misclassify far-range ramps as steps (underestimated $d^\text{xy}$) and under-penalise near-range ledges. Converting to Cartesian first ensures the pairwise slope is range-invariant.

The step score is then:

$$\tau[i,j] = \max_k(\delta z_k \mid k \text{ qualifies})\cdot\frac{|\{k : k \text{ qualifies}\}|}{|\mathcal{N}_5|-1}$$

This is a product of two terms: $h_\text{max} = \max_k(\delta z_k \mid k \text{ qualifies})$, the severity of the tallest confirmed step edge, and $f = |\{k : k \text{ qualifies}\}| / 24$, the fraction of neighbours that independently voted for a step. A genuine ledge produces a large $f$; a noise spike does not. The combined effect:

| Qualifying neighbours | $\tau$ |
|---|---|
| 1 of 24 | $\approx h_\text{max}/24$ — suppressed |
| 12 of 24 | $h_\text{max}/2$ — moderate |
| 24 of 24 | $h_\text{max}$ — full penalty |

$\tau$ is clipped to $\infty$ when $\tau > h_\text{crit}$.

**Step 3 — Combine into a danger score**

The three features are normalized by their critical thresholds and blended with fixed weights:

$$D[i,j] = 0.3\,\frac{\sigma}{\sigma_\text{crit}} + 0.3\,\frac{\rho}{\rho_\text{crit}} + 0.4\,\frac{\tau}{h_\text{crit}}$$

The higher weight on $\tau$ reflects that abrupt discontinuities are the most salient obstacle signal.

**Step 4 — Ray cast mask**

LiDAR returns only mark bins where photons landed, leaving unobserved gaps between the sensor and the furthest return in each angular column. Ray casting infers those gaps as free space using the occupancy-grid convention: a ray is free along its entire path up to the first hit.

For each angular column $j$, define the obstacle horizon as the nearest non-traversable radial bin:

$$R^*_j = \min\left\lbrace i : D[i,j] > d_\text{thresh}\right\rbrace$$

($R^*_j = \infty$ if no non-traversable cell exists in column $j$). The free-space mask is then:

$$M_\text{free}[i,j] = \mathbf{1}\!\left[i < R^*_j\right]$$

**Step 5 — Mask and threshold**

The three cases are combined into the final traversability map:

$$T[i,j] = \begin{cases} 1.0 & \text{if } \mathbf{1}_\text{valid}[i,j] \text{ and } D[i,j] > d_\text{thresh} \\ 0.0 & \text{if } M_\text{free}[i,j] \text{ and } D[i,j] \leq d_\text{thresh} \\ \text{NaN} & \text{otherwise (unobserved)} \end{cases}$$

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
