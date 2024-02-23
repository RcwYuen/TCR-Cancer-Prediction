import shutil
import os
from pathlib import Path

# Define the origin and destination directories
origin_dir = Path.cwd() / "data/compressed/control/"  # Update this to your actual origin directory
destination_dir = Path.cwd() / "data/trimmed/control/"  # Update this to your actual destination directory

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

paths = {}

for p in list(origin_dir.glob(r"dcr_HCW_*.tsv")):
    p_details = p.name.replace(".tsv", "").split("_")
    patient_id, timestamp = p_details[2], p_details[3].upper() == "BL"
    rpt = p_details[4] == "rpt"
    chain = p_details[-1]
    # print (p.name, patient_id, timestamp, rpt, chain)
    dict_key = (patient_id, chain)

    if timestamp and rpt:
        paths[dict_key] = p
        
    elif timestamp and (dict_key not in paths.keys()
                        or
                        paths[dict_key].name.replace(".tsv", "").split("_")[4] != "rpt"):
        paths[dict_key] = p

required = list(paths.values()) + list(origin_dir.glob(r"dcr_TCV_*"))
npatients = set()

for f in required:
    npatients.add(f.name.replace(".tsv", "").split("_")[2])
    target_path = destination_dir / f.name
    shutil.copy2(f, target_path)

print (f"There are {len(npatients)} patients") # 86