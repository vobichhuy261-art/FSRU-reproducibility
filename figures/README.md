# Portable manuscript figures

Install requirements.txt, then run `python figures/build_figures.py` from the
repository root. Use `--output PATH` to choose another output directory.
All input data are included in ../source_data; the original project directory
and author machine paths are not required. Outputs: editable PDF/SVG, 600-dpi
TIFF and 300-dpi PNG for each of the five figures. The panel-alignment check
runs before export with a 1.5-point tolerance. Rounded bootstrap values and
baseline summary match the manuscript figure labels. These figures reproduce
saved results; this program does not train models or redo the bootstrap.

build_figures.py is a portable adaptation of the author manuscript plotting
script. audit_panel_alignment.py is the nature-figure skill's bundled helper,
included as directed by that skill's Python backend instructions; it is a QA
utility, not part of the original FSRU implementation. No blanket license for
third-party material is asserted. Final exported PDFs still need text-size,
collision and visual review before use in a revised manuscript.
