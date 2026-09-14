# SLAI T-Rex project homepage

This directory is the self-contained static site published by GitHub Pages.

## Local preview

From the repository root:

```bash
python3 -m http.server 4173 --directory website
```

Open <http://127.0.0.1:4173/> in a browser. The site does not require a build step.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes this directory when
the `main` branch is updated. In the repository settings, set
**Pages -> Build and deployment -> Source** to **GitHub Actions**.
