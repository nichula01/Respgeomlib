# Figures

The page (`../../index.html`) references the figure images in this folder. Missing
files show a clean neutral placeholder box until the matching image is added. PNG,
JPG, or SVG all work.

## Current filenames

| Filename                             | Used in section                | Source / purpose                         |
|--------------------------------------|--------------------------------|------------------------------------------|
| `teaser.png`                         | Teaser                         | YAML to primitives to geometry to CFD     |
| `architecture.png`                   | Method                         | RespGeomLib architecture and pipeline     |
| `building_blocks.png`                | Method                         | Pipe, Y2, and Y3 building blocks          |
| `coordinate_convention.png`          | Additional Figures             | Local coordinate convention               |
| `ct_vs_procedural.png`               | Additional Figures             | CT-derived vs. RespGeomLib airway         |
| `synthetic_variants.png`             | Results                        | Baseline, stenosis, and dilation variants |
| `cfd.png`                            | Results                        | CFD demonstration                         |
| `residual_convergence.png`           | Results                        | Residual convergence                      |
| `weibel_like_model.png`              | Additional Figures / Demo      | Generated Weibel-like airway model        |

## Tip

To export figures straight from the paper PDF (one PNG per page) you can use:

```bash
pdftoppm -png -r 200 "RespGeomLib (3).pdf" page
```

then crop the panels you want into the filenames above.
