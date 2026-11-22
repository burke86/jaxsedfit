"""Generate the QSOGEN broad-line equivalent-width table shipped with jaxsedfit.

The table feeds ``agn.line_extrapolation_anchor = "continuum_ew"``: broad
emission lines outside the fitted spectral range are given a luminosity equal
to a rest-frame equivalent width times the fitted disc continuum.

Source: QSOGEN (Temple, Hewett & Banerji 2021, MNRAS 508, 737), MIT licence,
https://github.com/MJTemple/qsogen. The script reads the emission-line template
``qsosed_emlines_20210625.dat`` and the parameters of ``config.py`` from a local
checkout and reproduces the line-template mixing of
``Quasar_sed.add_emission_lines`` in its equivalent-width scaling mode
(``scal_emline < 0``), with the narrow-line template removed
(``scal_nlr = 0``) because jaxsedfit extrapolates narrow lines separately.

For every line group the rest equivalent width is

    EW = |scal_emline| * integral[(linval - nlr) / conval - pseudo] dlambda

over the group's integration window, where ``pseudo`` is a straight line
through the median of ``|scal_emline| (linval - nlr) / conval`` in a blue and
a red side band. The side bands remove the Fe II and Balmer-continuum
pseudo-continuum that QSOGEN carries inside the same template. The table is
tabulated on a grid of QSOGEN ``emline_type`` values; jaxsedfit interpolates
it at the ``emline_type`` implied by the fitted disc luminosity.

Usage::

    python tools/generate_qsogen_line_ew.py /path/to/qsogen \
        src/jaxsedfit/resources/templates/qsogen_line_ew.txt
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

import numpy as np


TEMPLATE_NAME = "qsosed_emlines_20210625.dat"

# name, blue side band, red side band, integration window. Groups that share a
# pseudo-continuum (C IV and the 1600 A He II + O III] complex) use the same
# side bands and adjacent integration windows. All wavelengths are rest-frame A.
LINE_GROUPS = (
    ("Lyb+OVI", (985.0, 995.0), (1070.0, 1090.0), (995.0, 1070.0)),
    ("Lya+NV", (1140.0, 1155.0), (1282.0, 1292.0), (1155.0, 1287.0)),
    ("OI+CII", (1282.0, 1292.0), (1348.0, 1356.0), (1287.0, 1352.0)),
    ("SiIV+OIV]", (1348.0, 1356.0), (1430.0, 1450.0), (1352.0, 1440.0)),
    ("CIV", (1430.0, 1450.0), (1680.0, 1695.0), (1440.0, 1595.0)),
    ("HeII+OIII]", (1430.0, 1450.0), (1680.0, 1695.0), (1595.0, 1687.0)),
    ("CIII]+AlIII+SiIII]", (1795.0, 1815.0), (1975.0, 1995.0), (1805.0, 1985.0)),
    ("MgII", (2650.0, 2680.0), (2900.0, 2930.0), (2665.0, 2915.0)),
    ("Hg", (4210.0, 4250.0), (4420.0, 4450.0), (4230.0, 4435.0)),
    # The red side band sits beyond the H-beta red wing and the Fe II 4924/5018
    # complex; a band at 4930-4950 A lies on the wing and halves the EW.
    ("Hb", (4700.0, 4750.0), (5080.0, 5110.0), (4725.0, 4940.0)),
    ("HeI5876", (5740.0, 5790.0), (5960.0, 6010.0), (5765.0, 5985.0)),
    ("Ha", (6150.0, 6200.0), (6850.0, 6900.0), (6175.0, 6875.0)),
    ("CaII+OI8446", (8340.0, 8380.0), (8700.0, 8740.0), (8360.0, 8720.0)),
    ("HeI10830+Pag", (10560.0, 10620.0), (11080.0, 11140.0), (10590.0, 11110.0)),
    ("Paa", (18400.0, 18490.0), (19030.0, 19120.0), (18445.0, 19075.0)),
)

# Narrow lines. "full": isolated forbidden lines measured in the full QSOGEN
# template (scal_nlr = 1) above a local linear pseudo-continuum, so their
# strength follows QSOGEN's emline_type mixing (the peaky/windy templates carry
# different [O III]). "nlr": lines on top of broad lines, integrated from the
# narrow-line (NLR) column alone (independent of emline_type). QSOGEN has no
# narrow UV or near-IR lines; jaxsedfit narrow lines outside these windows get
# zero EW.
NARROW_GROUPS = (
    ("[NeV]3426", "full", (3395.0, 3405.0), (3445.0, 3455.0), (3410.0, 3440.0)),
    ("[OII]3727", "full", (3695.0, 3710.0), (3750.0, 3765.0), (3712.0, 3745.0)),
    ("[NeIII]3869", "full", (3845.0, 3855.0), (3900.0, 3910.0), (3858.0, 3880.0)),
    ("Hg_n+[OIII]4363", "nlr", None, None, (4330.0, 4375.0)),
    ("HeII4686_n", "nlr", None, None, (4676.0, 4696.0)),
    ("Hb_n", "nlr", None, None, (4850.0, 4872.0)),
    ("[OIII]5007", "full", (4975.0, 4982.0), (5030.0, 5040.0), (4990.0, 5025.0)),
    # Fixed by atomic physics to [O III] 5007 / 2.98; its local pseudo-continuum
    # sits on the H-beta red wing and Fe II and is not measured reliably.
    ("[OIII]4959", "tie:[OIII]5007:2.98", None, None, (4945.0, 4972.0)),
    ("[FeVII]6087", "nlr", None, None, (6077.0, 6097.0)),
    ("[OI]6300", "nlr", None, None, (6288.0, 6312.0)),
    ("[FeX]6374", "nlr", None, None, (6362.0, 6382.0)),
    ("Ha_n", "nlr", None, None, (6553.0, 6572.0)),
    ("[NII]6583", "nlr", None, None, (6575.0, 6595.0)),
)

EMLINE_TYPE_GRID = np.round(np.arange(-2.0, 3.0 + 1.0e-9, 0.1), 6)


def read_qsogen_params(qsogen_dir: Path) -> dict[str, float]:
    """Read the scalar parameters used here from QSOGEN's ``config.py``."""
    tree = ast.parse((qsogen_dir / "config.py").read_text())
    wanted = {"scal_emline", "beslope", "benorm", "scal_lya", "scal_halpha"}
    values: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
            for keyword in node.keywords:
                if keyword.arg in wanted:
                    values[keyword.arg] = float(ast.literal_eval(keyword.value))
    missing = wanted - set(values)
    if missing:
        raise ValueError(f"QSOGEN config.py lacks parameters: {sorted(missing)}")
    return values


def qsogen_line_ratio(
    template: np.ndarray, emline_type: float, params: dict[str, float], scal_nlr: float = 0.0
) -> np.ndarray:
    """Return QSOGEN's broad-line flux per unit continuum, ``|s| linval / conval``.

    Follows ``Quasar_sed.add_emission_lines`` for ``scal_emline < 0`` with
    ``scal_nlr = 0``, including its clipping of negative dips and its
    ``scal_lya`` / ``scal_halpha`` multipliers.
    """
    linwav, medval, conval, pkyval, wdyval, nlr = template
    scanlr = float(scal_nlr)
    varlin = float(emline_type)
    if varlin == 0.0:
        linval = medval + (scanlr - 1.0) * nlr
    elif varlin > 0.0:
        varlin = min(varlin, 3.0)
        linval = varlin * pkyval + (1.0 - varlin) * medval + (scanlr - 1.0) * nlr
    else:
        varlin = min(abs(varlin), 2.0)
        linval = varlin * wdyval + (1.0 - varlin) * medval + (scanlr - 1.0) * nlr
    linval = np.array(linval, dtype=float)
    linval[(linwav > 4930) & (linwav < 5030) & (linval < 0.0)] = 0.0
    linval[(linwav > 1150) & (linwav < 1200) & (linval < 0.0)] = 0.0
    scale = np.full(linwav.shape, abs(params["scal_emline"]))
    scale[(linwav >= 6000.0) & (linwav <= 7000.0)] *= abs(params["scal_halpha"])
    scale[linwav < 1350.0] *= abs(params["scal_lya"])
    return scale * linval / conval


def _band_median(wave: np.ndarray, ratio: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
    sel = (wave >= band[0]) & (wave <= band[1])
    if np.count_nonzero(sel) < 2:
        raise ValueError(f"Side band {band} holds fewer than two template pixels.")
    return float(np.mean(band)), float(np.median(ratio[sel]))


def group_equivalent_width(wave: np.ndarray, ratio: np.ndarray, group) -> float:
    """Integrate one line group above its local linear pseudo-continuum."""
    _name, blue, red, window = group
    xb, yb = _band_median(wave, ratio, blue)
    xr, yr = _band_median(wave, ratio, red)
    lo, hi = window
    grid = np.linspace(lo, hi, 4001)
    excess = np.interp(grid, wave, ratio) - (yb + (yr - yb) * (grid - xb) / (xr - xb))
    return float(np.trapezoid(excess, grid))


def narrow_equivalent_width(template: np.ndarray, emline_type: float, params: dict[str, float], group) -> float:
    """Narrow-line rest EW for one narrow group (see NARROW_GROUPS)."""
    name, method, blue, red, (lo, hi) = group
    wave = template[0]
    if method == "nlr":
        nlr_ratio = abs(params["scal_emline"]) * template[5] / template[2]
        grid = np.linspace(lo, hi, 2001)
        return float(np.trapezoid(np.interp(grid, wave, nlr_ratio), grid))
    ratio = qsogen_line_ratio(template, emline_type, params, scal_nlr=1.0)
    return group_equivalent_width(wave, ratio, (name, blue, red, (lo, hi)))


def build_table(qsogen_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    template = np.genfromtxt(qsogen_dir / TEMPLATE_NAME, unpack=True)
    params = read_qsogen_params(qsogen_dir)
    ew = np.zeros((len(LINE_GROUPS), EMLINE_TYPE_GRID.size))
    narrow = np.zeros((len(NARROW_GROUPS), EMLINE_TYPE_GRID.size))
    for j, emline_type in enumerate(EMLINE_TYPE_GRID):
        ratio = qsogen_line_ratio(template, float(emline_type), params)
        for i, group in enumerate(LINE_GROUPS):
            ew[i, j] = group_equivalent_width(template[0], ratio, group)
        names = [group[0] for group in NARROW_GROUPS]
        for i, group in enumerate(NARROW_GROUPS):
            if group[1].startswith("tie:"):
                continue
            narrow[i, j] = max(narrow_equivalent_width(template, float(emline_type), params, group), 0.0)
        for i, group in enumerate(NARROW_GROUPS):
            if group[1].startswith("tie:"):
                _tie, target, divisor = group[1].split(":")
                narrow[i, j] = narrow[names.index(target), j] / float(divisor)
    return ew, narrow, params


def _qsogen_commit(qsogen_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(qsogen_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_table(path: Path, ew: np.ndarray, narrow: np.ndarray, params: dict[str, float], commit: str) -> None:
    header = [
        "# Broad emission-line rest equivalent widths [Angstrom] from QSOGEN.",
        "# Source: QSOGEN, Temple, Hewett & Banerji (2021), MNRAS 508, 737;",
        "#   https://github.com/MJTemple/qsogen (MIT licence, see LICENSES/QSOGEN-MIT.txt).",
        f"#   Template {TEMPLATE_NAME}, upstream commit {commit}.",
        "# Generated by tools/generate_qsogen_line_ew.py; do not edit by hand.",
        "# broad: EW = |scal_emline| * int[(linval - nlr)/conval - local linear pseudo-continuum] dlambda",
        "#   over [window_lo, window_hi]; linval mixes the median, peaky and windy templates",
        "#   following Quasar_sed.add_emission_lines with scal_nlr = 0.",
        "# narrow: isolated forbidden lines from the full template (scal_nlr = 1) above a local",
        "#   pseudo-continuum; lines on broad lines from |scal_emline| * int nlr/conval dlambda.",
        "# EWs are relative to QSOGEN's continuum (power law + hot-dust blackbody).",
        f"# scal_emline = {params['scal_emline']:.6g}; beslope = {params['beslope']:.6g}; "
        f"benorm = {params['benorm']:.6g}",
        "# emline_type = (M_i(z=2) - benorm) * beslope, clipped by QSOGEN to [-2, 3].",
        "# emline_type_grid: " + " ".join(f"{value:.1f}" for value in EMLINE_TYPE_GRID),
        "# columns: name family window_lo window_hi EW(emline_type_grid[0]) ... EW(emline_type_grid[-1])",
    ]
    lines = list(header)
    for group, row in zip(LINE_GROUPS, ew):
        name, _blue, _red, (lo, hi) = group
        values = " ".join(f"{value:.4f}" for value in row)
        lines.append(f"{name} broad {lo:.1f} {hi:.1f} {values}")
    for group, row in zip(NARROW_GROUPS, narrow):
        name, _method, _blue, _red, (lo, hi) = group
        values = " ".join(f"{value:.4f}" for value in row)
        lines.append(f"{name} narrow {lo:.1f} {hi:.1f} {values}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qsogen_dir", type=Path, help="Local QSOGEN checkout.")
    parser.add_argument("output", type=Path, help="Output table path.")
    args = parser.parse_args()
    ew, narrow, params = build_table(args.qsogen_dir)
    write_table(args.output, ew, narrow, params, _qsogen_commit(args.qsogen_dir))
    mid = int(np.argmin(np.abs(EMLINE_TYPE_GRID)))
    for group, row in zip(LINE_GROUPS, ew):
        print(f"{group[0]:>20s} broad  EW(emline_type=0) = {row[mid]:8.2f} A")
    for group, row in zip(NARROW_GROUPS, narrow):
        print(f"{group[0]:>20s} narrow EW(emline_type=0) = {row[mid]:8.2f} A")


if __name__ == "__main__":
    main()
