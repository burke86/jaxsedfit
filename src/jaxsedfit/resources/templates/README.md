# Template provenance

This directory contains vendored template files redistributed with `jaxsedfit`.

- Upstream project: `GRAHSP`
- Upstream license: `CeCILL v2`
- Local license copy: [CeCILL-v2.txt](../../../../LICENSES/CeCILL-v2.txt)

Vendored files:

- `Fe_d11-m20-20.5.txt`
  Source: `GRAHSP/database_builder/activate/agn/FeII_template/Fe_d11-m20-20.5.txt`
- `emission_line_table.formatted`
  Source: `GRAHSP/database_builder/activate/agn/mor_netzer_2012/emission_line_table.formatted`

The `dale2014/` subdirectory has its own provenance note.

`qsogen_line_ew.txt` is not from GRAHSP. It holds broad-line rest equivalent
widths derived from the QSOGEN emission-line template
(`qsosed_emlines_20210625.dat`, Temple, Hewett & Banerji 2021,
https://github.com/MJTemple/qsogen, MIT licence:
[QSOGEN-MIT.txt](../../../../LICENSES/QSOGEN-MIT.txt)) by
`tools/generate_qsogen_line_ew.py`.
