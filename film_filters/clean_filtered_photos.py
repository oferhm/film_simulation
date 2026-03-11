from pathlib import Path

# delete jpg in current folder
for f in Path(".").glob("*.jpg"):
    f.unlink()

# delete jpg in output folder
for f in Path("output").glob("*.jpg"):
    f.unlink()