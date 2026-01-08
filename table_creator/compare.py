import glob
import re
import matplotlib.pyplot as plt
from config import Variables

city = Variables.CITY_NORMAL
files = glob.glob(f"{city}/filtered_test/filtered-{city}-*.txt")

# Extract numeric C,H,F values from filename safely
def extract_params(filename):
    match = re.search(r"C([\d.]+)-H([\d.]+)-F([\d.]+)(?=\.txt)", filename)
    if match:
        C, H, F = map(float, match.groups())
        return C, H, F
    return None

# Load + parse + sort by C value
entries = []

for f in files:
    extracted = extract_params(f)
    if extracted:
        C, H, F = extracted
        with open(f) as file:
            ids = set(line.strip() for line in file)
        entries.append((C, ids))

# Sort by lowest → highest C
entries.sort(key=lambda x: x[0])

# Prepare data for plotting
c_values = [f"C={C:g}" for C, _ in entries]
counts = [len(ids) for _, ids in entries]

# ---- Plot bar chart ----
plt.figure(figsize=(10, 4))
bars = plt.bar(c_values, counts)

plt.xticks(rotation=0)
plt.ylabel("Number of images")
plt.title("Image count per C value")
plt.tight_layout()

# ---- Annotate bars with total count ----
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.show()