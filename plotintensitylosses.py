import matplotlib.pyplot as plt
import argparse
import pandas as pd

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.sans-serif': "Arial",
    'font.family': "sans-serif"
})

parser = argparse.ArgumentParser(description="Plot CSP data per spectrum.")
parser.add_argument("input_file", help="Path to input CSV file")
args = parser.parse_args()

df = pd.read_csv(args.input_file)  # Replace with actual file name

# Optional: Ensure types are correct
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['average'] = pd.to_numeric(df['average'], errors='coerce')

color_map = {
    'FL': '#F1A340',
    'NT': '#3856FF',
    'CT': '#AA2352'
}

# Plot each protein on the same figure
plt.figure(figsize=(8, 5))
for protein, sub_df in df.groupby('protein'):
    color = color_map.get(protein, 'gray')  # Default to gray if not found
    plt.plot(sub_df['ratio'], sub_df['average'], marker='o', label=protein, color=color)

# Plot formatting
plt.title("Average Intensity vs Molar Ratio for FL, NT, CT")
plt.xlabel("Molar Ratio (Protein:Substrate)")
plt.ylabel("Average Intensity Ratio")
# plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
