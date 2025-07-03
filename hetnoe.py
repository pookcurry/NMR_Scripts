import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from collections import defaultdict
import argparse
import re
import pandas as pd

# --- Matplotlib Setup ---
def configure_matplotlib():
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.sans-serif': "Arial",
        'font.family': "sans-serif"
    })

# --- Input Setup ---
def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSP data per spectrum.")
    parser.add_argument("file1", help="Path to first dataset file")
    parser.add_argument("-f2", "--file2", default=None, help="Optional path to second dataset file")    
    parser.add_argument("-seq", "--sequence_file", type=str, help="1-letter amino acid sequence (e.g., matloop.txt)")
    parser.add_argument("-psi", "--psipred_file", help="PSIPRED .ss2 file for secondary structure overlay")
    parser.add_argument("-xlim", "--x_limit", type=float, default=None, help="Override the X-axis limit (residue number)")
    parser.add_argument("-ylim", "--y_limit", type=float, default=None, help="Override the Y-axis limit (CSP or IL Ratio)")
    parser.add_argument("-il", "--intensity_loss", action="store_true", help="Enable to plot intensity losses instead of CSPs")
    parser.add_argument("-apo", "--apo_spectrum", type=str, help="Explicitly specify the name of the apo spectrum \n(could be useful if there are issues with intensity)")
    parser.add_argument("-s", "--save", action="store_true", help="Enable to save plots as .svg files")
    parser.add_argument("-skip", "--skip_nterm", type=int, default=0, help='Number of N-terminal residues to skip in plots (default: 0)')
    parser.add_argument("-sf", "--sofast", action="store_true", help="Enable to plot SOFAST data per residue")
    return parser.parse_args()

# --- Helper Functions ---
def sanitise_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace illegal characters with underscores

def read_sequence(filepath):
    with open(filepath, 'r') as file:
        seq =  file.read().strip().replace('\n', '')
    return seq

def parse_ss2(filepath):
    structure_map = {'H': 'helix', 'E': 'beta', 'C': 'coil'}
    ss_seq = []
    residues = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            try:
                res_id = int(parts[0])
                ss_type = structure_map.get(parts[2], 'coil')
                ss_seq.append((res_id, ss_type))
                residues.append(res_id)
            except (IndexError, ValueError):
                continue  # skip malformed lines

    # Group contiguous secondary structures
    blocks = []
    current = None
    for res_id, ss in ss_seq:
        if current is None or ss != current['type']:
            if current: blocks.append(current)
            current = {'type': ss, 'start': res_id, 'end': res_id}
        else:
            current['end'] = res_id
    if current: blocks.append(current)

    return blocks, max(residues)

def read_data(filepath):
    """Read titration CSV and return structured list of entries."""
    df = pd.read_csv(filepath)
    original_df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.iloc[:, [3, 4, 7, 8, 12, 13, 14, 17]]
    df.columns = ['series_step_x', 'additional_series_step_x', 'ppm', 
                  'height', 'res', 'name', 'atom', 'spectrum']

    for col in ['series_step_x', 'additional_series_step_x', 'ppm', 'height', 'res']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[df['name'].notna()]

    molar_ratio = {}
    unique_specs = df[['spectrum', 'series_step_x', 'additional_series_step_x']].drop_duplicates()

    for _, row in unique_specs.iterrows():
        spectrum = row['spectrum']
        stepx = row['series_step_x']
        addx = row['additional_series_step_x']

        if addx and addx != 0:
            molar_ratio[spectrum] = stepx / addx
        else:
            molar_ratio[spectrum] = 0  # or maybe np.nan or None
            print(f"WARNING: Invalid additional_series_step_x for spectrum {spectrum}")

    ns_dict = {}

    if original_df.shape[1] > 25:
        try:
            ns_series = pd.to_numeric(original_df.iloc[:, 25], errors='coerce')
            df['ns'] = ns_series
            ns_dict = df. groupby('spectrum')['ns'].mean().to_dict()

        except Exception:
            pass
        
    residues_present = sorted(df['res'].dropna().unique().astype(int))

    # Prioritize series_step_x == 0, otherwise pick the lowest series_step_x > 0
    apo_candidates = df[df['series_step_x'].notna()].sort_values(by='series_step_x')
    if any(apo_candidates['series_step_x'] == 0):
        apo_spectrum = apo_candidates[apo_candidates['series_step_x'] == 0].iloc[0]['spectrum']
    else:
        apo_spectrum = apo_candidates.iloc[0]['spectrum']

    baseline_noise = {}

    if True:
        try:
            # Convert the baseline noise column (column 24) to numeric values
            noise_series = pd.to_numeric(original_df.iloc[:, 24], errors='coerce')
            
            # Add the noise values to the DataFrame
            df['noise'] = noise_series
            
            # Group by 'spectrum' and calculate the mean baseline noise for each spectrum
            baseline_noise = df.groupby('spectrum')['noise'].mean().to_dict()

        except Exception as e:
            # Print out the error message if something goes wrong
            print(f"Error processing baseline noise: {e}")

    return df.to_dict(orient='records'), apo_spectrum, residues_present, ns_dict, molar_ratio, baseline_noise

def organise_by_spectrum(data, value_key):
    """Organise shift data by spectrum -> residue -> atom."""
    data_spec = defaultdict(lambda: defaultdict(dict))

    for entry in data:
        if entry['spectrum'] and value_key in entry:
            data_spec[entry['spectrum']][entry['res']][entry['atom']] = entry[value_key]

    return data_spec

def calculate_height(data_spec, apo, ns_dict, baseline_noise):
    comparisons = {}
    noise_comparisons = {}
    spectra = set(data_spec) - {apo}

    # ns_values = [value for value in ns_dict.values() if value > 0]
    # min_ns = min(ns_values) if ns_values else 1

    ns_apo = ns_dict.get(apo, 1)

    for spec in spectra:
        ns = ns_dict.get(spec, 1)
        scale = (ns / ns_apo) if ns > 0 else 1

        for res in data_spec[spec].values():
            if 'H' in res and res['H'] is not None:
                res['H_scaled'] = res['H'] / scale
    
    for res in data_spec[apo].values():
        if 'H' in res and res['H'] is not None:
            res['H_scaled'] = res['H'] 

    for spec in spectra:
        heights = []
        errors = []

        for res in data_spec[apo]:
            if res > 0 and res in data_spec[spec]:
                h_apo = data_spec[apo][res].get('H_scaled')
                h_other = data_spec[spec][res].get('H_scaled')

                if h_apo and h_other:
                    ratio = h_other / h_apo if h_apo != 0 else 0

                    noise_apo = baseline_noise.get(apo, 0)
                    noise_other = baseline_noise.get(spec, 0)

                    delta_h_apo = noise_apo / h_apo if h_apo else 0
                    delta_h_other = noise_other / h_other if h_other else 0

                    delta_ratio = ratio * np.sqrt(delta_h_apo**2 + delta_h_other**2)

                    heights.append({'Residue': res, 'Ratio': ratio})
                    errors.append({'Residue': res, 'Error': delta_ratio})

        comparisons[spec] = heights
        noise_comparisons[spec] = errors

    return comparisons, noise_comparisons

def get_optimal_font_size(ax, sequence, buffer=1.1, max_font=10, min_font=5):
    """Estimate a monospace font size that prevents overlap."""
    n_residues = len(sequence)
    
    # Get axis width in display units (points)
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in_inches = bbox.width
    dpi = fig.dpi
    width_in_points = width_in_inches * dpi

    # Estimate space per character
    space_per_char = width_in_points / n_residues

    # Estimate optimal font size (monospace, so ~1:1 width:height in points)
    est_font_size = space_per_char * buffer

    return max(min(est_font_size, max_font), min_font)

def map_sequence(ax, residues_present, sequence, skip_nterm=0):
    optimal_font = get_optimal_font_size(ax, sequence)
    ax.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=False, labelbottom=True)
    for i, aa in enumerate(sequence):
        if i < skip_nterm:
            continue
        residue_index = i + 1
        ax.text(
            i + 1, 1.025, aa, 
            ha='center', va='bottom',
            fontsize=optimal_font,
            fontweight='bold' if residue_index in residues_present else 'normal',
            family='monospace', fontname='Courier New',
            transform=ax.get_xaxis_transform()
        )

def plot_secondary_structure(ax, filepath, skip_nterm=0):
    blocks, max_resid = parse_ss2(filepath)

    ss_y = 1.1  # Adjust height above x-axis

    for ss in blocks:
        x_start = ss["start"]
        x_end = ss["end"] if ss["type"] in ["helix", "beta"] else max_resid

        if x_end < skip_nterm:
            continue  # Entire block is before the visible region

        # Clip block start if it begins before skip_nterm
        x_start = max(x_start, skip_nterm+1)
        length = x_end - x_start + 1

        ax.add_patch(
            plt.Rectangle(
                (x_start - 0.5, ss_y + 0.012),
                length,
                0.005,
                color='black',
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=1
            )
        )
        
        if ss["type"] == "helix":
            ax.add_patch(
                plt.Rectangle(
                    (x_start - 0.5, ss_y),
                    length,
                    0.02,
                    color='white',
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=2
                )
            )
            
            x_vals = np.linspace(0, 1, 200)
            wave_length = x_end - x_start + 1
            num_waves = wave_length / 2
            x_wave = (x_vals * wave_length) + (x_start - 0.5)
            y_wave = ss_y + 0.015 * np.sin(2 * np.pi * num_waves * x_vals)
            ax.plot(
                x_wave, y_wave + 0.015,
                color='skyblue',
                linewidth = 2.5,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=3
            )

        elif ss["type"] == "beta":
            ax.add_patch(
                FancyArrow(
                    x_start - 0.5,
                    ss_y + 0.015,
                    length,
                    0,
                    width=0.03,
                    head_width=0.06,
                    head_length=0.5,  # you can tweak for style
                    length_includes_head=True,
                    color='gold',
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=2
                )
            )

def collect_intensity_data(data, intensities_per_comparison, noise_per_comparison):
    """Collect spectrum number, intensities and errors for each residue, averaging if duplicates."""
    residue_intensity_data = defaultdict(list)
    residue_error_data = defaultdict(list)

    # Make data index for looking up series_step_x
    data_index = {(entry['spectrum'], entry['res']): entry for entry in data if entry['series_step_x'] is not None}

    for spectrum in intensities_per_comparison:
        intensities = intensities_per_comparison[spectrum]
        errors = noise_per_comparison.get(spectrum, [])

        # Make a quick lookup for errors by residue
        error_lookup = {e['Residue']: e['Error'] for e in errors}

        for result in intensities:
            res = result['Residue']
            intensity = result['Ratio']
            error = error_lookup.get(res, None)

            entry = data_index.get((spectrum, res))
            if entry is None:
                continue

            spectrum_number = entry['series_step_x']

            # Handle averaging of duplicates
            existing_intensities = [v for t, v in residue_intensity_data[res] if t == spectrum_number]
            existing_errors = [v for t, v in residue_error_data[res] if t == spectrum_number]

            if existing_intensities:
                average_intensity = np.mean(existing_intensities + [intensity])
            else:
                average_intensity = intensity

            if error is not None:
                if existing_errors:
                    average_error = np.mean(existing_errors + [error])
                else:
                    average_error = error
            else:
                average_error = None

            # Remove existing entries for this spectrum_number
            residue_intensity_data[res] = [(t, v) for t, v in residue_intensity_data[res] if t != spectrum_number]
            residue_intensity_data[res].append((spectrum_number, average_intensity))

            if average_error is not None:
                residue_error_data[res] = [(t, v) for t, v in residue_error_data[res] if t != spectrum_number]
                residue_error_data[res].append((spectrum_number, average_error))

    return residue_intensity_data, residue_error_data

def compile_intensity_data(intensity_data1, error_data1, intensity_data2=None, error_data2=None):
    res_ids_1 = []
    hetnoe_values_1 = []
    hetnoe_errors_1 = []

    for res in sorted(intensity_data1.keys()):
        value = intensity_data1[res]
        if isinstance(value, list):
            ratio = np.mean([v for _, v in value])
        else:
            ratio = value

        res_ids_1.append(res)
        hetnoe_values_1.append(ratio)

        if error_data1 and res in error_data1:
            errs = error_data1[res]
            if isinstance(errs, list):
                error_val = np.mean([v for _, v in errs])
            else:
                error_val = errs
            hetnoe_errors_1.append(error_val)
        else:
            hetnoe_errors_1.append(None)

    res_ids_2 = []
    hetnoe_values_2 = []
    hetnoe_errors_2 = []

    if intensity_data2:
        for res in sorted(intensity_data2.keys()):
            value = intensity_data2[res]
            if isinstance(value, list):
                ratio = np.mean([v for _, v in value])
            else:
                ratio = value

            res_ids_2.append(res)
            hetnoe_values_2.append(ratio)

            if error_data2 and res in error_data2:
                errs = error_data2[res]
                if isinstance(errs, list):
                    error_val = np.mean([v for _, v in errs])
                else:
                    error_val = errs
                hetnoe_errors_2.append(error_val)
            else:
                hetnoe_errors_2.append(None)

    return {
        "res_ids_1": res_ids_1,
        "hetnoe_values_1": hetnoe_values_1,
        "hetnoe_errors_1": hetnoe_errors_1,
        "res_ids_2": res_ids_2,
        "hetnoe_values_2": hetnoe_values_2,
        "hetnoe_errors_2": hetnoe_errors_2
    }

def plot_hetnoe(compiled, sequence=None, psipred_file=None, y_limit=None, x_limit=None, save=False, skip_nterm=0):
    res_ids_1 = compiled['res_ids_1']
    values_1 = compiled['hetnoe_values_1']
    errors_1 = compiled['hetnoe_errors_1']  
    res_ids_2 = compiled['res_ids_2']
    values_2 = compiled['hetnoe_values_2']
    errors_2 = compiled['hetnoe_errors_2']

    global_max = max(values_1 + values_2) if (values_1 and values_2) else 1
    ymax = y_limit if y_limit else min(np.ceil((global_max + 1e-6) / 1) * 1, global_max)

    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)

    if values_1:
        ax.errorbar(
            res_ids_1, values_1, yerr=errors_1,
            fmt='-o', color='tab:blue', capsize=3, markersize=4,
            label='LABEL1'
        )

    if values_2:
        ax.errorbar(
            res_ids_2, values_2, yerr=errors_2,
            fmt='-o', color='tab:red', capsize=3, markersize=4,
            label='LABEL2'
        )

    ax.set_ylim(-0.5, ymax * 2)
    ax.set_xlim(skip_nterm, x_limit if x_limit else max(res_ids_1) + 1)

    if sequence:
        map_sequence(ax, res_ids_1 + res_ids_2, sequence, skip_nterm)
    if psipred_file:
        plot_secondary_structure(ax, psipred_file, skip_nterm)

    ax.set_xlabel('Residue')
    ax.set_ylabel(r'HetNOE ($I_{\mathrm{unsat}}$ / $I_{\mathrm{sat}}$)')
    if sequence:
        tick_positions = np.arange(0, len(sequence), 10)
        ax.set_xticks(tick_positions)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x - skip_nterm)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_title(fr"HetNOEs per Residue", y=1.16 if sequence and psipred_file else 1.08 if sequence else 1)
    ax.legend()
    plt.tight_layout()

    if save:
        name = sanitise_filename("HetNOE_summary")
        plt.savefig(f"{name}_HetNOE.svg")
    else:
        plt.show()

# --- Main Execution ---
def main():
    configure_matplotlib()
    args = parse_args()

    sequence = read_sequence(args.sequence_file) if args.sequence_file else None
    if not sequence:
        print("\n⏩ Skipping sequence plotting. \nUse '-seq' or '--sequence_file' with a .txt of the sequence to overlay above the plot.")
    if not args.psipred_file:
        print("\n⏩ Skipping secondary structure plotting. \nUse '-psi' or '--psipred_file' with a .ss2 from PSIPRED to overlay above the plot.")

    data1, apo1, _, scans1, _, baseline_noise_1 = read_data(args.file1)
    # id_to_name_mapping1 = {entry['res']: entry['name'] for entry in data1 if entry['res'] is not None}
    apo1 = args.apo_spectrum if args.apo_spectrum else apo1

    data_spec1 = organise_by_spectrum(data1, value_key='height')
    comparisons1, noise_comparisons1 = calculate_height(data_spec1, apo1, scans1, baseline_noise_1)
    res_intensity_data1, res_error_data1 = collect_intensity_data(data1, comparisons1, noise_comparisons1)

    res_intensity_data2 = None
    res_intensity_data2 = None
    res_error_data2 = None
    # id_to_name_mapping2 = None

    if args.file2:
        data2, apo2, _, scans2, _, baseline_noise_2 = read_data(args.file2)
        # id_to_name_mapping2 = {entry['res']: entry['name'] for entry in data2 if entry['res'] is not None}
        data_spec2 = organise_by_spectrum(data2, value_key='height')
        comparisons2, noise_comparisons2 = calculate_height(data_spec2, apo2, scans2, baseline_noise_2)
        res_intensity_data2, res_error_data2 = collect_intensity_data(data2, comparisons2, noise_comparisons2)

    compiled_data = compile_intensity_data(res_intensity_data1, res_error_data1, res_intensity_data2, res_error_data2)

    plot_hetnoe(
        compiled_data,
        sequence=sequence,
        psipred_file=args.psipred_file,
        y_limit=args.y_limit if hasattr(args, 'y_limit') else None,
        x_limit=args.x_limit if hasattr(args, 'x_limit') else None,
        save=args.save if hasattr(args, 'save') else False,
        skip_nterm=args.skip_nterm if hasattr(args, 'skip_nterm') else 0
    )

if __name__ == "__main__":
    main()
