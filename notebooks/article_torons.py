import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import topovec as tv

    return Path, mo, np, tv


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    REPO_DIR = NOTEBOOK_DIR.parent
    TMP_DIR = REPO_DIR / "tmp"
    return (TMP_DIR,)


@app.cell
def _(mo):
    def display_image(image, caption=None):
        return mo.image(image, caption=caption)

    return (display_image,)


@app.cell
def _(display_image, mo, np, tv):
    def physical_grid(size_um=(60.0, 60.0, 10.0), step_um=0.5):
        size_um = np.asarray(size_um, dtype=np.float64)
        counts = tuple(int(round(length / step_um)) + 1 for length in size_um)
        system = tv.System.cubic(size=counts)
        xyz = system.spin_positions().astype(np.float64)
        center = (np.asarray(counts, dtype=np.float64) - 1.0) / 2.0
        xyz = (xyz - center[None, None, None, None, :]) * step_um
        return system, xyz

    def render_layer_image(
        system,
        nn: np.ndarray,
        *,
        axis: int,
        layer: int,
        imgsize: int = 1024,
        width: float = 0.5,
        max_size: float = 0.9,
        mesh: str = "Cylinder",
        show_axes: bool = False,
    ):
        ic = tv.mgl.render_layer(
            system,
            axis=axis,
            layer=layer,
            imgsize=imgsize,
            mesh=mesh,
            show_axes=show_axes,
        )
        ic.upload(nn)
        ic.scene["Cones"]["Max. size"] = max_size
        ic.scene["Cones"]["Width"] = width
        return ic.save()

    def render_two_slice_panels(
        system,
        nn: np.ndarray,
        *,
        label: str,
        xy_layer: int | None = None,
        xz_layer: int | None = None,
        imgsize: int = 1024,
        width: float = 0.5,
        max_size: float = 0.9,
        thin_step: int = 3,
    ):
        if thin_step == 1:
            thin_nn = nn
            thin_system = system
        else:
            thin_nn, thin_system = system.thinned(data=nn, steps=thin_step)
        if xy_layer is None:
            xy_layer = thin_nn.shape[2] // 2
        if xz_layer is None:
            xz_layer = thin_nn.shape[1] // 2
        xy_layer = int(xy_layer)
        xz_layer = int(xz_layer)
        xz_image = render_layer_image(
            thin_system,
            thin_nn,
            axis=1,
            layer=xz_layer,
            imgsize=imgsize,
            width=width,
            max_size=max_size,
        )
        xy_image = render_layer_image(
            thin_system,
            thin_nn,
            axis=2,
            layer=xy_layer,
            imgsize=imgsize,
            width=width,
            max_size=max_size,
        )
        return mo.vstack(
            [
                display_image(xz_image, caption=f"{label}: xz slice @ y-index {xz_layer}"),
                display_image(xy_image, caption=f"{label}: xy slice @ z-index {xy_layer}"),
            ],
            align="start",
            justify="start",
        )

    def normalize_field(nn: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), eps)

    def cone_basis_article(xyz: np.ndarray, angle_rad: float, tol: float = 1e-32):
        x = np.asarray(xyz[..., 0])
        y = np.asarray(xyz[..., 1])
        z = np.asarray(xyz[..., 2])
        rho = np.sqrt(np.maximum(x * x + y * y, tol))
        x0 = x / rho
        y0 = y / rho
        cc = np.cos(angle_rad)
        ss = np.sin(angle_rad)
        dx = np.stack([x0 * cc, y0 * cc, -np.ones_like(x0) * ss], axis=-1)
        dy = np.stack([-y0, x0, np.zeros_like(x0)], axis=-1)
        dz = np.stack([x0 * ss, y0 * ss, np.ones_like(x0) * cc], axis=-1)
        a = cc * z + ss * rho
        return a, dx, dy, dz

    def anticone_basis_article(xyz: np.ndarray, angle_rad: float, tol: float = 1e-32):
        x = np.asarray(xyz[..., 0])
        y = np.asarray(xyz[..., 1])
        z = np.asarray(xyz[..., 2])
        rho = -np.maximum(np.sqrt(x * x + y * y), tol)
        x0 = x / rho
        y0 = y / rho
        cc = np.cos(angle_rad)
        ss = np.sin(angle_rad)
        dx = np.stack([x0 * cc, y0 * cc, -np.ones_like(x0) * ss], axis=-1)
        dy = np.stack([-y0, x0, np.zeros_like(x0)], axis=-1)
        dz = np.stack([x0 * ss, y0 * ss, np.ones_like(x0) * cc], axis=-1)
        a = cc * z + ss * rho
        return a, dx, dy, dz

    def spiral_article(a, dx, dy, dz, *, period_um: float, shift_um: float):
        phase = ((a + shift_um) * 2.0 * np.pi / period_um)[..., None]
        return dx * np.cos(phase) + dy * np.sin(phase)

    def outer_cone_uniform_article(distance, *, slope: float, shift_um: float):
        field = np.zeros(distance.shape + (3,), dtype=np.float64)
        field[..., 2] = np.exp(slope * (distance - shift_um))
        return field

    def surface_field_article(
        xyz: np.ndarray,
        *,
        exponent: float = -2.0,
        scale: float = 1.1,
        guard: float = 1e-8,
    ):
        z = np.asarray(xyz[..., 2], dtype=np.float64)
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        field = np.zeros(xyz.shape[:-1] + (3,), dtype=np.float64)
        field[..., 2] = (
            ((z - z_min) / scale + guard) ** exponent
            + ((z_max - z) / scale + guard) ** exponent
        )
        return field

    def build_s1_field(xyz_um: np.ndarray, size_um):
        width_um = float(size_um[2])
        pitch_um = 15.0
        arm_um = 14.0
        outer_radius_um = 7.0
        shift0_um = 0.0
        cut_slope = 0.7
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        angle_rad = float(np.arctan2(width_um, arm_um))
        outer_shift_um = width_um * outer_radius_um / w_um

        a0, dx0, dy0, dz0 = cone_basis_article(xyz_um, angle_rad)
        a1, dx1, dy1, dz1 = anticone_basis_article(xyz_um, angle_rad)
        a2, _, _, _ = cone_basis_article(xyz_um, -angle_rad)
        nn = (
            spiral_article(a0, dx0, dy0, dz0, period_um=pitch_um, shift_um=shift0_um)
            + spiral_article(a1, dx1, dy1, dz1, period_um=pitch_um, shift_um=shift0_um)
            + outer_cone_uniform_article(a2, slope=-cut_slope, shift_um=-outer_shift_um)
            + outer_cone_uniform_article(a0, slope=cut_slope, shift_um=outer_shift_um)
            + surface_field_article(xyz_um)
        )
        return normalize_field(nn)

    def build_s2_field(xyz_um: np.ndarray, size_um):
        width_um = float(size_um[2])
        pitch_um = 12.0
        arm_um = 11.0
        outer_radius_um = 10.0
        outer_height_um = 0.0
        shift0_um = 2.5
        cut_slope = 0.7
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        angle_rad = float(np.arctan2(width_um, arm_um))
        outer_shift_front_um = (width_um * outer_radius_um + arm_um * outer_height_um) / w_um
        outer_shift_back_um = (width_um * outer_radius_um - arm_um * outer_height_um) / w_um

        a0, dx0, dy0, dz0 = cone_basis_article(xyz_um, angle_rad)
        a1, dx1, dy1, dz1 = anticone_basis_article(xyz_um, angle_rad)
        a2, _, _, _ = cone_basis_article(xyz_um, -angle_rad)
        rho = np.sqrt(xyz_um[..., 0] ** 2 + xyz_um[..., 1] ** 2)[..., None]
        nn = (
            spiral_article(a0, dx0, dy0, dz0, period_um=pitch_um, shift_um=shift0_um)
            * np.exp(-0.5 * rho)
            + spiral_article(a1, dx1, dy1, dz1, period_um=pitch_um, shift_um=shift0_um)
            + outer_cone_uniform_article(a2, slope=-cut_slope, shift_um=-outer_shift_front_um)
            + outer_cone_uniform_article(a0, slope=cut_slope, shift_um=outer_shift_back_um)
            + surface_field_article(xyz_um)
        )
        return normalize_field(nn)

    def build_s3_field(xyz_um: np.ndarray, size_um):
        width_um = float(size_um[2])
        pitch_um = 12.0
        arm_um = 14.0
        outer_radius_um = 23.0
        inner_radius_um = 5.0
        outer_height_um = 0.0
        inner_height_um = 0.0
        shift0_um = -3.0
        cut_slope = 0.7
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        angle_rad = float(np.arctan2(width_um, arm_um))
        outer_shift_front_um = (width_um * outer_radius_um + arm_um * outer_height_um) / w_um
        outer_shift_back_um = (width_um * outer_radius_um - arm_um * outer_height_um) / w_um
        inner_shift_front_um = (width_um * inner_radius_um + arm_um * inner_height_um) / w_um
        inner_shift_back_um = (width_um * inner_radius_um - arm_um * inner_height_um) / w_um

        a1, dx1, dy1, dz1 = anticone_basis_article(xyz_um, angle_rad)
        a0, _, _, _ = cone_basis_article(xyz_um, -angle_rad)
        a2, _, _, _ = cone_basis_article(xyz_um, angle_rad)
        nn = (
            spiral_article(a1, dx1, dy1, dz1, period_um=pitch_um, shift_um=shift0_um)
            + outer_cone_uniform_article(a0, slope=-cut_slope, shift_um=-outer_shift_front_um)
            + outer_cone_uniform_article(a0, slope=cut_slope, shift_um=-inner_shift_front_um)
            + outer_cone_uniform_article(a2, slope=cut_slope, shift_um=outer_shift_back_um)
            + outer_cone_uniform_article(a2, slope=-cut_slope, shift_um=inner_shift_back_um)
            + surface_field_article(xyz_um)
        )
        return normalize_field(nn)

    def step_label(step_um: float) -> str:
        text = f"{step_um:.3f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    return (
        build_s1_field,
        build_s2_field,
        build_s3_field,
        physical_grid,
        render_two_slice_panels,
        step_label,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Direct article-formula torons

    This notebook reproduces the explicit `S1`, `S2`, and `S3` fields from
    the article formulas on the original article grid.
    """)
    return


@app.cell
def _():
    size_um = (60.0, 60.0, 10.0)
    step_um = 0.5
    return size_um, step_um


@app.cell
def _(physical_grid, size_um, step_um):
    system, xyz_um = physical_grid(size_um=size_um, step_um=step_um)
    return system, xyz_um


@app.cell(hide_code=True)
def _(mo, size_um, step_um, system):
    mo.md(f"""
    Physical box: `{size_um}` um

    Grid spacing: `{step_um}` um

    Grid size: `{tuple(int(value) for value in system.size)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S1

    Direct article formula for the `S1` toron.
    """)
    return


@app.cell
def _(build_s1_field, size_um, xyz_um):
    s1_nn = build_s1_field(xyz_um[..., 0, :], size_um)[..., None, :]
    return (s1_nn,)


@app.cell
def _(render_two_slice_panels, s1_nn, system):
    render_two_slice_panels(system, s1_nn, label="S1", thin_step=3)
    return


@app.cell
def _(s1_nn, tv):
    tv.marimo.inspect(s1_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s1_output_path = TMP_DIR / f"article_toron_S1_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s1_output_path, s1_nn, settings={"comment": "Generated by article_torons.py", "L": 10.0})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S2

    Direct article formula for the `S2` toron.
    """)
    return


@app.cell
def _(build_s2_field, size_um, xyz_um):
    s2_nn = build_s2_field(xyz_um[..., 0, :], size_um)[..., None, :]
    return (s2_nn,)


@app.cell
def _(render_two_slice_panels, s2_nn, system):
    render_two_slice_panels(system, s2_nn, label="S2", thin_step=3)
    return


@app.cell
def _():
    # tv.marimo.inspect(s2_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s2_output_path = TMP_DIR / f"article_toron_S2_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s2_output_path, s2_nn, settings={"comment": "Generated by article_torons.py", "L": 10.0})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S3

    Direct article formula for the `S3` toron.
    """)
    return


@app.cell
def _(build_s3_field, size_um, xyz_um):
    s3_nn = build_s3_field(xyz_um[..., 0, :], size_um)[..., None, :]
    return (s3_nn,)


@app.cell
def _(render_two_slice_panels, s3_nn, system):
    render_two_slice_panels(system, s3_nn, label="S3", thin_step=3)
    return


@app.cell
def _():
    # tv.marimo.inspect(s3_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s3_output_path = TMP_DIR / f"article_toron_S3_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s3_output_path, s3_nn, settings={"comment": "Generated by article_torons.py", "L": 10.0})
    return


if __name__ == "__main__":
    app.run()
