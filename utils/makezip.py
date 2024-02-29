import zipfile
import glob

paths = [
    "data/sceptr-traintest/**",
]

files = []
for path in paths:
    files = files + list(glob.glob(path, recursive = True))

with zipfile.ZipFile("upload.zip", "w") as zipf:
    for f in files:
        print (f"Zipping {f}")
        zipf.write(f)