# Handling Large PDF Files

Place course PDFs inside `Dev/data/` which is ignored by Git. They are not versioned by default.

If repository limits become a problem later, you can retroactively move PDFs into Git LFS with:

```bash
git lfs migrate import '*.pdf'
```

Otherwise keep the files locally and back them up to Google Drive or GCS as needed.
