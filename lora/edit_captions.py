from pathlib import Path
import random

# Load the captions
captions_path = Path("/u/mpamnani/lora/coco_text_extended/caption.txt")
with captions_path.open("r") as f:
    captions = [line.strip() for line in f if line.strip()]

# Define a pool of random prefixes
random_prefixes = [
    "scribble", "write", "engrave", "print", 
    "sketch", "inscribe", "display"
]

# Transform each line to the new format
modified_captions = []
for word in captions:
    prefix = random.choice(random_prefixes)
    modified_line = f"{prefix} <txter> {word}"
    modified_captions.append(modified_line)

# Save to a new file
output_path = Path("modified_captions.txt")
with output_path.open("w") as f:
    for line in modified_captions:
        f.write(f"{line}\n")

output_path.name
