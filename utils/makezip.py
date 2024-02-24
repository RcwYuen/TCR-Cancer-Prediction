import zipfile
import glob

paths = [
    "data/trimmed/**",
    # "model/mlm-only/model/**",
    # "model/mlm-only/tokenizer/**",
    # "scripts/**",
    # "utils/**",
    # "trainer.py",
    # "model.py"
]
files = []
for path in paths:
    files = files + list(glob.glob(path, recursive = True))

with zipfile.ZipFile("upload.zip", "w") as zipf:
    for f in files:
        print (f"Zipping {f}")
        zipf.write(f)