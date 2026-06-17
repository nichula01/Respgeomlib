# RespGeomLib — Project Website

A single-page **classic academic project page** for the IEEE MERCon 2026 paper
*“RespGeomLib: A Reproducible Parametric Engine for Generating Analysis-Ready Human
Airway Lumen Geometry.”* Styled after Nerfies / Zip-NeRF-type CVPR/MICCAI project pages:
centered title block, teaser figure, abstract, method, results tables, compact additional
figures, and citation — calm, minimal, professor-facing.

## Files

```
website/
├── index.html        # the page (semantic HTML, ~960px centered)
├── styles.css        # classic academic styling
├── script.js         # graceful figure placeholders + copy-BibTeX (no animation libs)
├── assets/figures/   # real paper figures go here (see its README)
├── assets/models/    # STL and GLB files for the interactive 3D preview
└── tools/            # helper scripts for website assets
```

## Run locally

Static site — open `index.html`, or serve it:

```bash
cd website
python -m http.server 8080   # then open http://localhost:8080
```

## What to add

1. **Paper PDF** — the `Paper` button points to `paper.pdf` in this folder:
   ```bash
   cp "RespGeomLib (3).pdf" website/paper.pdf
   ```
2. **Figures** — the page references the images in `assets/figures/`. Missing files
   show a clean placeholder box until you drop them in. See
   `assets/figures/README.md` for the current filenames.

## 3D preview models

The interactive geometry preview uses GLB files in `assets/models/`. To regenerate
them from the STL files:

```bash
python tools/convert_stl_to_glb.py
```

## Customize

- Colors / fonts: the `:root` variables at the top of `styles.css`.
- GitHub link: `https://github.com/nichula01/RespGeomLib` (update in `index.html` if it changes).
- Status is stated as **“Accepted at IEEE MERCon 2026”** (not “published”).

No build step. Fonts load from Google Fonts and the equation is rendered with MathJax.
Deploy anywhere static (GitHub Pages, Netlify, Vercel) by publishing this folder.
