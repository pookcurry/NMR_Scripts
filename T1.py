import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from collections import defaultdict
import argparse
import re
import pandas as pd
import math
from scipy.optimize import curve_fit

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


    return df.to_dict(orient='records'), apo_spectrum, residues_present, ns_dict, molar_ratio

def organise_by_spectrum(data, value_key):
    """Organise shift data by spectrum -> residue -> atom."""
    data_spec = defaultdict(lambda: defaultdict(dict))

    for entry in data:
        if entry['spectrum'] and value_key in entry:
            data_spec[entry['spectrum']][entry['res']][entry['atom']] = entry[value_key]

    return data_spec

def calculate_height(data_spec, apo, ns_dict):
    comparisons = {}
    spectra = set(data_spec) - {apo}

    # ns_values = [value for value in ns_dict.values() if value > 0]
    # min_ns = min(ns_values) if ns_values else 1

    ns_apo = ns_dict.get(apo, 1)

    for spec in spectra:
        ns = ns_dict.get(spec, 1)
        scale = (ns / ns_apo)**1 if ns > 0 else 1

        for res in data_spec[spec].values():
            if 'H' in res and res['H'] is not None:
                res['H_scaled'] = res['H'] / scale
    
    for res in data_spec[apo].values():
        if 'H' in res and res['H'] is not None:
            res['H_scaled'] = res['H'] 

    for spec in spectra:
        heights = []

        for res in data_spec[apo]:
            if res > 0 and res in data_spec[spec]:
                h_apo = data_spec[apo][res].get('H_scaled')
                h_other = data_spec[spec][res].get('H_scaled')

                if h_apo and h_other:
                    ratio = h_other / h_apo if h_apo != 0 else 0
                    heights.append({'Residue': res, 'Ratio': ratio})

        comparisons[spec] = heights

    return comparisons

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

def collect_intensity_data(data, intensities_per_comparison):
    """Collect times and intensities for each residue, averaging intensity for duplicate time points."""
    residue_intensity_data = defaultdict(list)
    data_index = {(entry['spectrum'], entry['res']): entry for entry in data if entry['series_step_x'] is not None}

    for spectrum, intensities in intensities_per_comparison.items():
        for result in intensities:
            res = result['Residue']
            intensity = result['Ratio']
            entry = data_index.get((spectrum, res))

            if entry:
                time_point = entry['series_step_x']
                existing = residue_intensity_data[res]

                # Check if this time_point already exists
                existing_times = [tp for tp, _ in existing]
                if time_point in existing_times:
                    # Get all existing intensities for this time point
                    new_data = [(tp, val) for tp, val in existing if tp == time_point] + [(time_point, intensity)]
                    # Remove all existing entries for this time_point
                    residue_intensity_data[res] = [(tp, val) for tp, val in existing if tp != time_point]
                    # Add the averaged one back
                    avg_intensity = np.mean([val for _, val in new_data])
                    residue_intensity_data[res].append((time_point, avg_intensity))
                else:
                    # No duplicate: just add
                    residue_intensity_data[res].append((time_point, intensity))

    return residue_intensity_data

def t1_model(t, I0, T1):
    return I0 * np.exp(-t / T1)

def fit_t1(data1, name_map1, data2=None, name_map2=None):
    t1_values_1, t1_errors_1, res_ids_1 = [], [], []
    t1_values_2, t1_errors_2, res_ids_2 = [], [], []

    # If two datasets are provided, use common residues
    if data2:
        residues = sorted(set(data1) & set(data2))
    else:
        residues = sorted(data1)

    num_residues = len(residues)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)

    # Create subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2.5), constrained_layout=True)
    axs = axs.flatten()

    for i, res in enumerate(residues):
        try:
            ax = axs[i]  # Get the current subplot axis
            t_fit = None

            # Dataset 1
            if res in data1:
                pts1 = data1[res][:]

                # Ensure you add the (50, 1) point if it's missing
                if not any(t == 1.0 for t, _ in pts1):
                    pts1.append((50, 1))
                pts1 = sorted(pts1)

                if len(pts1) >= 3:
                    t1, i1 = zip(*pts1)
                    if all(i > 0 for i in i1):
                        popt1, pcov1 = curve_fit(t1_model, t1, i1, p0=(max(i1), 20))
                        t_fit = np.linspace(min(t1), max(t1), 100)
                        # Calculate the error (standard deviation) for T1 in Dataset 1
                        stddev_T1_1 = np.sqrt(np.diag(pcov1))[1]
                        # Add both T1 and its error to the legend
                        ax.plot(t_fit, t1_model(t_fit, *popt1), '-', color='tab:blue', 
                                label=f'T1={popt1[1]:.2f} \n       ± {stddev_T1_1:.2f}')
                        
                        # Dataset 1
                        t1_values_1.append(popt1[1])
                        t1_errors_1.append(stddev_T1_1)
                        res_ids_1.append(res)


            # Dataset 2
            if data2 and res in data2:
                pts2 = data2[res][:]

                # Ensure you add the (50, 1) point if it's missing
                if not any(t == 1.0 for t, _ in pts2):
                    pts2.append((50, 1))
                pts2 = sorted(pts2)

                if len(pts2) >= 3:
                    t2, i2 = zip(*pts2)
                    if all(i > 0 for i in i2):
                        popt2, pcov2 = curve_fit(t1_model, t2, i2, p0=(max(i2), 20))
                        if t_fit is None:
                            t_fit = np.linspace(min(t2), max(t2), 100)
                        # Calculate the error (standard deviation) for T1 in Dataset 2
                        stddev_T1_2 = np.sqrt(np.diag(pcov2))[1]
                        # Add both T1 and its error to the legend
                        ax.plot(t_fit, t1_model(t_fit, *popt2), '-', color='tab:red', 
                                label=f'T1={popt2[1]:.2f} \n       ± {stddev_T1_2:.2f}')

                        # Dataset 2
                        t1_values_2.append(popt2[1])
                        t1_errors_2.append(stddev_T1_2)
                        res_ids_2.append(res)

            if t_fit is not None:
                # Set title, labels, and other plot details
                name = name_map1.get(res) or (name_map2.get(res) if name_map2 else f"Res {res}")
                ax.set_title(f'{name} {res}', fontsize=8)
                ax.set_xlabel('τ (ms)', fontsize=7)
                if i % ncols == 0:
                    ax.set_ylabel("Intensity Ratio", fontsize=7)

                # Add the legend with T1 and the error
                ax.legend(loc='best', fontsize=6)

                # ax.set_xlim(-5, 2005)
                ax.set_xticks(np.linspace(0, 2000, 3))
                ax.set_ylim(-0.05, 1.05)

        except Exception as e:
            print(f"⚠️ T1 fit failed for residue {res}: {e}")
            axs[i].axis('off')  # If error occurs, hide the subplot for this residue

    # Hide any unused subplots
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.suptitle(fr"$T_1$ Fits per Residue", fontsize=14)

    plt.show()

    return {
    "res_ids_1": res_ids_1,
    "t1_values_1": t1_values_1,
    "t1_errors_1": t1_errors_1,
    "res_ids_2": res_ids_2,
    "t1_values_2": t1_values_2,
    "t1_errors_2": t1_errors_2
}

def plot_t1(fit_result, sequence=None, psipred_file=None, y_limit=None, x_limit=None, save=False, skip_nterm=0):
    res_ids_1 = fit_result['res_ids_1']
    values_1 = fit_result['t1_values_1']
    errors_1 = fit_result['t1_errors_1']
    res_ids_2 = fit_result['res_ids_2']
    values_2 = fit_result['t1_values_2']
    errors_2 = fit_result['t1_errors_2']

    all_values = values_1 + values_2
    global_max = max(all_values) if all_values else 0
    ymax = y_limit if y_limit else min(np.ceil((global_max + 1e-6) / 50) * 50, global_max * 1.3)

    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)

    if values_1:
        ax.plot(res_ids_1, values_1, '-o', color='tab:blue', label='LABELb')
        ax.errorbar(
        res_ids_1, values_1, yerr=errors_1,
        fmt='o', color='tab:blue', capsize=3, markersize=4
    )

    if values_2:
        ax.plot(res_ids_2, values_2, '-o', color='tab:red', label='LABELa')
        ax.errorbar(
        res_ids_2, values_2, yerr=errors_2,
        fmt='o', color='tab:red', capsize=3, markersize=4
    )

    ax.set_ylim(0, ymax * 1.3)
    ax.set_xlim(skip_nterm, x_limit if x_limit else max(res_ids_1 + res_ids_2) + 1)

    if sequence:
        map_sequence(ax, res_ids_1 + res_ids_2, sequence, skip_nterm)
    if psipred_file:
        plot_secondary_structure(ax, psipred_file, skip_nterm)

    ax.set_xlabel('Residue')
    ax.set_ylabel(fr'$T_1$ (ms)')
    if sequence:
        tick_positions = np.arange(0, len(sequence), 10)
        ax.set_xticks(tick_positions)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x - skip_nterm)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_title(fr"$T_1$ Relaxation Times per Residue", y=1.16 if sequence and psipred_file else 1.08 if sequence else 1)
    ax.legend()
    plt.tight_layout()

    if save:
        name = sanitise_filename("T1_summary")
        plt.savefig(f"{name}_T1.svg")
    else:
        plt.show()

def calculate_r1(t1_values, t1_errors):
    r1_values = (1 / np.array(t1_values)) * 1000
    r1_errors = (np.array(t1_errors) / (np.array(t1_values) ** 2)) * 1000
    return r1_values, r1_errors

def plot_r1(fit_result, sequence=None, psipred_file=None, y_limit=None, x_limit=None, save=False, skip_nterm=0):
    res_ids_1 = fit_result['res_ids_1']
    values_1 = fit_result['t1_values_1']
    errors_1 = fit_result['t1_errors_1']
    res_ids_2 = fit_result['res_ids_2']
    values_2 = fit_result['t1_values_2']
    errors_2 = fit_result['t1_errors_2']

    # Get R1 values and errors
    r1_values_1, r1_errors_1 = calculate_r1(values_1, errors_1)
    r1_values_2, r1_errors_2 = calculate_r1(values_2, errors_2)

    # Combine both datasets' R1 values to determine global max for y-axis scaling
    all_values = np.concatenate([r1_values_1, r1_values_2])
    global_max = np.max(all_values) if np.any(all_values) else 0
    ymax = y_limit if y_limit else min(np.ceil((global_max + 1e-6) / 50) * 50, global_max * 1.3)

    # Create the plot
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)

    # Plot dataset 1 (if data exists)
    if len(r1_values_1) > 0:
        ax.plot(res_ids_1, r1_values_1, '-o', color='tab:blue', label='LABEL1')
        ax.errorbar(res_ids_1, r1_values_1, yerr=r1_errors_1, fmt='o', color='tab:blue', capsize=3, markersize=4)

    # Plot dataset 2 (if data exists)
    if len(r1_values_2) > 0:
        ax.plot(res_ids_2, r1_values_2, '-o', color='tab:red', label='LABEL2')
        ax.errorbar(res_ids_2, r1_values_2, yerr=r1_errors_2, fmt='o', color='tab:red', capsize=3, markersize=4)

    # Set axis limits
    ax.set_ylim(0, ymax)
    ax.set_xlim(skip_nterm, x_limit if x_limit else max(res_ids_1 + res_ids_2) + 1)

    # Optionally plot sequence or psi prediction data
    if sequence:
        map_sequence(ax, res_ids_1 + res_ids_2, sequence, skip_nterm)
    if psipred_file:
        plot_secondary_structure(ax, psipred_file, skip_nterm)

    # Set labels and title
    ax.set_xlabel('Residue')
    ax.set_ylabel(r'$R_1(\mathrm{s}^{-1})$')
    if sequence:
        tick_positions = np.arange(0, len(sequence), 10)
        ax.set_xticks(tick_positions)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x - skip_nterm)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_title(fr"$R_1$ Relaxation Rates per Residue", y=1.16 if sequence and psipred_file else 1.08 if sequence else 1)
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if save:
        name = sanitise_filename("R1_summary")
        plt.savefig(f"{name}_R1.svg")
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

    data1, apo1, _, scans1, _ = read_data(args.file1)
    id_to_name_mapping1 = {entry['res']: entry['name'] for entry in data1 if entry['res'] is not None}
    apo1 = args.apo_spectrum if args.apo_spectrum else apo1

    data_spec1 = organise_by_spectrum(data1, value_key='height')
    comparisons1 = calculate_height(data_spec1, apo1, scans1)
    residue_intensity_data1 = collect_intensity_data(data1, comparisons1)

    residue_intensity_data2 = None
    id_to_name_mapping2 = None

    if args.file2:
        data2, apo2, _, scans2, _ = read_data(args.file2)
        id_to_name_mapping2 = {entry['res']: entry['name'] for entry in data2 if entry['res'] is not None}
        data_spec2 = organise_by_spectrum(data2, value_key='height')
        comparisons2 = calculate_height(data_spec2, apo2, scans2)
        residue_intensity_data2 = collect_intensity_data(data2, comparisons2)

    fit_result = fit_t1(residue_intensity_data1, id_to_name_mapping1, residue_intensity_data2, id_to_name_mapping2)

    plot_t1(
        fit_result,
        sequence=sequence,
        psipred_file=args.psipred_file,
        y_limit=args.y_limit if hasattr(args, 'y_limit') else None,
        x_limit=args.x_limit if hasattr(args, 'x_limit') else None,
        save=args.save if hasattr(args, 'save') else False,
        skip_nterm=args.skip_nterm if hasattr(args, 'skip_nterm') else 0
    )

    plot_r1(
        fit_result,
        sequence=sequence,
        psipred_file=args.psipred_file,
        y_limit=args.y_limit if hasattr(args, 'y_limit') else None,
        x_limit=args.x_limit if hasattr(args, 'x_limit') else None,
        save=args.save if hasattr(args, 'save') else False,
        skip_nterm=args.skip_nterm if hasattr(args, 'skip_nterm') else 0
    )

if __name__ == "__main__":
    main()
