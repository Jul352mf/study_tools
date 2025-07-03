import shutil
from pathlib import Path

# Paths to remove
TO_DELETE = [
    Path("chroma"),            # vector DB
    Path("cache"),             # all partials + modules
    Path("summary.md"),        # final markdown summary
    Path("summary.pdf"),       # optional PDF output
    Path("storage"),           # legacy folder, just in case
]

for path in TO_DELETE:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
            print(f"🧹 Deleted folder: {path}")
        else:
            path.unlink()
            print(f"🧹 Deleted file:   {path}")
    else:
        print(f"✅ Already clean:  {path}")

print("\n✨ Reset complete. Ready for a fresh start!")