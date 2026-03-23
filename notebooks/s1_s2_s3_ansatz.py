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
        max_size: float = 1.,
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

    def cone_coordinates(
        angle_rad: float,
        *,
        shift_um: tuple[float, float] = (0.0, 0.0),
        tol: float = 1e-12,
    ):
        meridional = tv.ansatz.EuclideanCoordinates(ndim=2).shift(shift_um).rotate(-angle_rad)
        return meridional.axisymmetrize(axis=2, tol=tol, orientation=1.0)

    def anticone_coordinates(
        angle_rad: float,
        *,
        shift_um: tuple[float, float] = (0.0, 0.0),
        tol: float = 1e-12,
    ):
        meridional = (
            tv.ansatz.EuclideanCoordinates(ndim=2)
            .shift(shift_um)
            .rotate(angle_rad)
            .mix([[-1.0, 0.0], [0.0, 1.0]])
        )
        return meridional.axisymmetrize(axis=2, tol=tol, orientation=-1.0)

    def cholesteric_spiral(
        coordinates,
        *,
        period_um: float,
        shift_um: float,
        inv_rho_mask,
    ):
        plus = tv.ansatz.CholestericSpiral3D(
            phase_axis=1,
            plane_axes=(0, 2),
            period=period_um,
            shift=shift_um,
            coordinates=coordinates,
        )
        minus = tv.ansatz.CholestericSpiral3D(
            phase_axis=1,
            plane_axes=(0, 2),
            period=-period_um,
            shift=shift_um,
            coordinates=coordinates,
        )
        cos_part = 0.5 * (plus + minus)
        sin_part = 0.5 * (plus + (-minus)) * inv_rho_mask
        return cos_part + sin_part

    def uniform_z_field():
        ambient = tv.ansatz.EuclideanCoordinates(ndim=3)
        return tv.ansatz.ConstantField3D((0.0, 0.0, 1.0), coordinates=ambient)

    def inv_rho_mask():
        ambient = tv.ansatz.EuclideanCoordinates(ndim=3)
        return ambient.mask(lambda x, y, z: 1.0 / np.maximum(np.sqrt(x * x + y * y), 1e-32))

    def radial_envelope(decay: float):
        ambient = tv.ansatz.EuclideanCoordinates(ndim=3)
        return ambient.mask(lambda x, y, z: np.exp(-decay * np.sqrt(x * x + y * y)))

    def surface_mask(size_um, *, exponent: float = -2.0, scale: float = 1.1, guard: float = 1e-8):
        half_thickness = 0.5 * float(size_um[2])
        ambient = tv.ansatz.EuclideanCoordinates(ndim=3)
        return ambient.mask(
            lambda x, y, z: ((z + half_thickness) / scale + guard) ** exponent
            + ((half_thickness - z) / scale + guard) ** exponent
        )

    def cut_mask(coordinates, *, slope: float, shift_um: float):
        return coordinates.mask(lambda b, a, theta: np.exp(slope * (a - shift_um)))

    def build_s1_ansatz(
        size_um,
        *,
        pitch_um: float = 15.0,
        arm_um: float = 14.0,
        outer_radius_um: float = 7.0,
        shift0_um: float = 0.0,
        cut_slope: float = 0.7,
    ):
        width_um = float(size_um[2])
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        outer_shift_um = width_um * outer_radius_um / w_um
        angle_rad = float(np.arctan2(width_um, arm_um))

        z_axis = uniform_z_field()
        inv_rho = inv_rho_mask()
        field = (
            cholesteric_spiral(
                cone_coordinates(angle_rad),
                period_um=pitch_um,
                shift_um=shift0_um,
                inv_rho_mask=inv_rho,
            )
            + cholesteric_spiral(
                anticone_coordinates(angle_rad),
                period_um=pitch_um,
                shift_um=shift0_um,
                inv_rho_mask=inv_rho,
            )
            + z_axis * cut_mask(cone_coordinates(-angle_rad), slope=-cut_slope, shift_um=-outer_shift_um)
            + z_axis * cut_mask(cone_coordinates(angle_rad), slope=cut_slope, shift_um=outer_shift_um)
            + z_axis * surface_mask(size_um)
        )
        return field

    def build_s2_ansatz(
        size_um,
        *,
        pitch_um: float = 12.0,
        arm_um: float = 11.0,
        outer_radius_um: float = 10.0,
        outer_height_um: float = 0.0,
        shift0_um: float = 2.5,
        cut_slope: float = 0.7,
        radial_decay: float = 0.5,
    ):
        width_um = float(size_um[2])
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        outer_shift_front_um = (width_um * outer_radius_um + arm_um * outer_height_um) / w_um
        outer_shift_back_um = (width_um * outer_radius_um - arm_um * outer_height_um) / w_um
        angle_rad = float(np.arctan2(width_um, arm_um))

        z_axis = uniform_z_field()
        inv_rho = inv_rho_mask()
        field = (
            cholesteric_spiral(
                cone_coordinates(angle_rad),
                period_um=pitch_um,
                shift_um=shift0_um,
                inv_rho_mask=inv_rho,
            )
            * radial_envelope(radial_decay)
            + cholesteric_spiral(
                anticone_coordinates(angle_rad),
                period_um=pitch_um,
                shift_um=shift0_um,
                inv_rho_mask=inv_rho,
            )
            + z_axis * cut_mask(cone_coordinates(-angle_rad), slope=-cut_slope, shift_um=-outer_shift_front_um)
            + z_axis * cut_mask(cone_coordinates(angle_rad), slope=cut_slope, shift_um=outer_shift_back_um)
            + z_axis * surface_mask(size_um)
        )
        return field

    def build_s3_ansatz(
        size_um,
        *,
        pitch_um: float = 12.0,
        arm_um: float = 14.0,
        outer_radius_um: float = 23.0,
        inner_radius_um: float = 5.0,
        outer_height_um: float = 0.0,
        inner_height_um: float = 0.0,
        shift0_um: float = -3.0,
        cut_slope: float = 0.7,
    ):
        width_um = float(size_um[2])
        w_um = float(np.sqrt(width_um**2 + arm_um**2))
        outer_shift_front_um = (width_um * outer_radius_um + arm_um * outer_height_um) / w_um
        outer_shift_back_um = (width_um * outer_radius_um - arm_um * outer_height_um) / w_um
        inner_shift_front_um = (width_um * inner_radius_um + arm_um * inner_height_um) / w_um
        inner_shift_back_um = (width_um * inner_radius_um - arm_um * inner_height_um) / w_um
        angle_rad = float(np.arctan2(width_um, arm_um))

        z_axis = uniform_z_field()
        inv_rho = inv_rho_mask()
        field = (
            cholesteric_spiral(
                anticone_coordinates(angle_rad),
                period_um=pitch_um,
                shift_um=shift0_um,
                inv_rho_mask=inv_rho,
            )
            + z_axis * cut_mask(cone_coordinates(-angle_rad), slope=-cut_slope, shift_um=-outer_shift_front_um)
            + z_axis * cut_mask(cone_coordinates(-angle_rad), slope=cut_slope, shift_um=-inner_shift_front_um)
            + z_axis * cut_mask(cone_coordinates(angle_rad), slope=cut_slope, shift_um=outer_shift_back_um)
            + z_axis * cut_mask(cone_coordinates(angle_rad), slope=-cut_slope, shift_um=inner_shift_back_um)
            + z_axis * surface_mask(size_um)
        )
        return field

    def step_label(step_um: float) -> str:
        text = f"{step_um:.3f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    def fixed_label(value: float, decimals: int = 1) -> str:
        return f"{float(value):.{int(decimals)}f}".replace(".", "p")

    return (
        build_s1_ansatz,
        build_s2_ansatz,
        build_s3_ansatz,
        physical_grid,
        render_two_slice_panels,
        step_label,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Torons from `topovec.ansatz`

    This notebook builds toron-like structures directly with the current
    coordinate-first `topovec.ansatz` API.

    The construction is:

    1. build cone and anti-cone coordinate charts from shifted/rotated meridional coordinates,
    2. form cholesteric spirals in those charts,
    3. add explicit cut and surface masks in physical units,
    4. normalize on a `0.5 um` cubic lattice,
    5. materialize ansatz-based `S1`, `S2`, and `S3`,
    6. supplement them with `CF1` loop demos on a separate hopfion-style grid.

    The explicit article formulas live in the separate notebook
    `notebooks/article_torons.py`.
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

    Two compensating spirals, two outer cuts, and surface anchoring.
    """)
    return


@app.cell
def _(build_s1_ansatz, size_um, xyz_um):
    s1_field = build_s1_ansatz(size_um)
    s1_nn = s1_field(xyz_um)
    return (s1_nn,)


@app.cell
def _(render_two_slice_panels, s1_nn, system):
    render_two_slice_panels(system, s1_nn, label="S1", thin_step=3)
    return


@app.cell
def _(s1_nn, tv):
    # tv.marimo.inspect(s1_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s1_output_path = TMP_DIR / f"toron_S1_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s1_output_path, s1_nn, settings={"comment": "Generated by s1_s2_s3_ansatz.py", "L": 10.0})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S2

    The front spiral gets an additional radial envelope, while the cut
    shifts follow the article `S2` parameters.
    """)
    return


@app.cell
def _(build_s2_ansatz, size_um, xyz_um):
    s2_field = build_s2_ansatz(size_um)
    s2_nn = s2_field(xyz_um)
    return (s2_nn,)


@app.cell
def _(render_two_slice_panels, s2_nn, system):
    render_two_slice_panels(system, s2_nn, label="S2", thin_step=3)
    return


@app.cell
def _(s2_nn, tv):
    # tv.marimo.inspect(s2_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s2_output_path = TMP_DIR / f"toron_S2_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s2_output_path, s2_nn, settings={"comment": "Generated by s1_s2_s3_ansatz.py", "L": 10.0})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## S3

    A single anti-cone spiral plus four exponential cut fields and surface
    anchoring.
    """)
    return


@app.cell
def _(build_s3_ansatz, size_um, xyz_um):
    s3_field = build_s3_ansatz(size_um)
    s3_nn = s3_field(xyz_um)
    return (s3_nn,)


@app.cell
def _(render_two_slice_panels, s3_nn, system):
    render_two_slice_panels(system, s3_nn, label="S3", thin_step=3)
    return


@app.cell
def _(s3_nn, tv):
    # tv.marimo.inspect(s3_nn)
    return


@app.cell
def _(TMP_DIR, step_label, step_um):
    s3_output_path = TMP_DIR / f"toron_S3_{step_label(step_um)}um.npz"
    # tv.io.save_npz_lcsim(s3_output_path, s3_nn, settings={"comment": "Generated by s1_s2_s3_ansatz.py", "L": 10.0})
    return


if __name__ == "__main__":
    app.run()
