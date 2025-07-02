import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import re
import pandas as pd
import math

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
    relative_uncertainty_comparisons = {}
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
        relative_uncertainty_errors = []

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
                    relative_uncertainty_added = (delta_h_apo + delta_h_other) / np.log(h_apo / h_other)

                    heights.append({'Residue': res, 'Ratio': ratio})
                    errors.append({'Residue': res, 'Error': delta_ratio})
                    relative_uncertainty_errors.append({'Residue': res, 'Relative_Error': relative_uncertainty_added})

        comparisons[spec] = heights
        noise_comparisons[spec] = errors
        relative_uncertainty_comparisons[spec] = relative_uncertainty_errors

    return comparisons, noise_comparisons, relative_uncertainty_comparisons

def collect_intensity_data(data, intensities_per_comparison, noise_per_comparison, relative_uncertainty_per_comparison):
    """Collect spectrum number, intensities and errors for each residue, averaging if duplicates."""
    residue_intensity_data = defaultdict(list)
    residue_error_data = defaultdict(list)
    residue_relative_uncertainty_data = defaultdict(list)  # Store relative uncertainties separately

    # Make data index for looking up series_step_x
    data_index = {(entry['spectrum'], entry['res']): entry for entry in data if entry['series_step_x'] is not None}

    for spectrum in intensities_per_comparison:
        intensities = intensities_per_comparison[spectrum]
        errors = noise_per_comparison.get(spectrum, [])
        relative_errors = relative_uncertainty_per_comparison.get(spectrum, [])

        # Make a quick lookup for errors by residue
        error_lookup = {e['Residue']: e['Error'] for e in errors}
        relative_error_lookup = {e['Residue']: e['Relative_Error'] for e in relative_errors}

        for result in intensities:
            res = result['Residue']
            intensity = result['Ratio']
            error = error_lookup.get(res, None)
            relative_error = relative_error_lookup.get(res, None)

            entry = data_index.get((spectrum, res))
            if entry is None:
                continue

            spectrum_number = entry['series_step_x']

            # Handle averaging of duplicates
            existing_intensities = [v for t, v in residue_intensity_data[res] if t == spectrum_number]
            existing_errors = [v for t, v in residue_error_data[res] if t == spectrum_number]
            existing_relative_errors = [v for t, v in residue_relative_uncertainty_data[res] if t == spectrum_number]

            # Averaging intensities
            if existing_intensities:
                average_intensity = np.mean(existing_intensities + [intensity])
            else:
                average_intensity = intensity

            # Averaging errors (standard)
            if error is not None:
                if existing_errors:
                    average_error = np.mean(existing_errors + [error])
                else:
                    average_error = error
            else:
                average_error = None

            # Averaging relative errors (uncertainty)
            if relative_error is not None:
                if existing_relative_errors:
                    average_relative_error = np.mean(existing_relative_errors + [relative_error])
                else:
                    average_relative_error = relative_error
            else:
                average_relative_error = None

            # Remove existing entries for this spectrum_number
            residue_intensity_data[res] = [(t, v) for t, v in residue_intensity_data[res] if t != spectrum_number]
            residue_intensity_data[res].append((spectrum_number, average_intensity))

            if average_error is not None:
                residue_error_data[res] = [(t, v) for t, v in residue_error_data[res] if t != spectrum_number]
                residue_error_data[res].append((spectrum_number, average_error))

            # Store the relative uncertainty separately
            if average_relative_error is not None:
                residue_relative_uncertainty_data[res] = [(t, v) for t, v in residue_relative_uncertainty_data[res] if t != spectrum_number]
                residue_relative_uncertainty_data[res].append((spectrum_number, average_relative_error))

    return residue_intensity_data, residue_error_data, residue_relative_uncertainty_data

def cpmg_model(t, I0, T2):
    return I0 * np.exp(-t / T2)

def extract_tau(data: list[dict]) -> float | None:
    """Get the relaxation delay (tau) from metadata."""
    # Extract unique values from the 'additional_series_step_x' key across all entries
    t_values = {entry['additional_series_step_x'] for entry in data if entry['additional_series_step_x'] is not None}
    
    if len(t_values) == 1:
        return t_values.pop()
    
    print("\n⚠️ Multiple or invalid tau values found; cannot determine global tau.")
    return None

def plot_cpmg_intensity_ratios(tau, res_intensity_data1, res_error_data1, name_map1, res_intensity_data2=None, res_error_data2=None, name_map2=None):
    """Plot measured I/I0 ratios vs echo number for each residue with error bars."""

    # Identify residues that are present in both datasets (if both data1 and data2 are provided)
    if res_intensity_data2:
        residues = sorted(set(res_intensity_data1) & set(res_intensity_data2))
    else:
        residues = sorted(res_intensity_data1)

    num_residues = len(residues)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 2.5),
        constrained_layout=True
    )
    axs = axs.flatten()

    for i, res in enumerate(residues):
        ax = axs[i]
        plotted_any = False

        # --- Dataset 1 ---  
        if res in res_intensity_data1:
            pts1 = sorted(res_intensity_data1[res])  # Extract the intensity data for the residue
            n1, i_over_i0_1 = zip(*pts1)
            error1_res = [e for t, e in res_error_data1.get(res, []) if t in n1]  # Get the corresponding errors
            ax.errorbar(n1, i_over_i0_1, yerr=error1_res, fmt='-o', color='tab:blue', capsize=1.5, markersize=4, label=name_map1.get(res, f"Res {res}"))
            plotted_any = True

        # --- Dataset 2 ---
        if res_intensity_data2 and res in res_intensity_data2:
            pts2 = sorted(res_intensity_data2[res])  # Extract the intensity data for the residue
            n2, i_over_i0_2 = zip(*pts2)
            error2_res = [e for t, e in res_error_data2.get(res, []) if t in n2]  # Get the corresponding errors
            ax.errorbar(n2, i_over_i0_2, yerr=error2_res, fmt='-o', color='tab:red', capsize=1.5, markersize=4, label=name_map2.get(res, f"Res {res}"))
            plotted_any = True

        if not plotted_any:
            ax.axis('off')
            continue

        # --- Axis Titles, Labels, Limits ---
        name = name_map1.get(res) or (name_map2.get(res) if name_map2 else f"Res {res}")
        ax.set_title(f'{name} {res}', fontsize=8)
        ax.set_xlabel('Echo Number (n)', fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel('Intensity Ratio', fontsize=7)

        # Legend
        # ax.legend(fontsize=6)

        # X limits
        ax.set_xlim(-5, 65)
        ax.set_xticks(np.linspace(0, 60, 3))

        # Y limits: Auto but padded
        all_y = []
        if res in res_intensity_data1:
            all_y.extend(i_over_i0_1)
        if res_intensity_data2 and res in res_intensity_data2:
            all_y.extend(i_over_i0_2)
        all_y = [y for y in all_y if np.isfinite(y)]
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            ypad = 3 * (ymax - ymin) if ymax != ymin else 0.05
            ax.set_ylim(ymin - ypad, ymax + ypad)
        else:
            ax.set_ylim(0, 1.2)

    # Hide unused subplots
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.suptitle(fr"CPMG I/I0 vs Echo Number per Residue (fixed τ = {tau} ms)", fontsize=14)
    plt.show()

def T2_eff_model(intensity_ratios, tau):
    return -tau / np.log(intensity_ratios)

def calc_T2_per_point(intensity_ratios, tau, relative_uncertainty):
    """Given a list of intensity ratios I/I0 and relative uncertainties, return the corresponding T2 values (ms) and errors."""

    intensity_ratios = np.array(intensity_ratios)

    with np.errstate(divide='ignore', invalid='ignore'):
        T2_eff = T2_eff_model(intensity_ratios, tau)
        T2_eff = np.where(intensity_ratios > 0, T2_eff, np.nan)
    
    T2_errors = T2_eff * np.array(relative_uncertainty)

    return T2_eff, T2_errors

def plot_T2_per_point(tau, data1, error1, name_map1, data2=None, error2=None, name_map2=None):
    """Plot T2 per point vs echo number (n) for all residues, using fixed total relaxation delay."""

    tmin = 0
    tmax = 1000

    if data2:
        residues = sorted(set(data1) & set(data2))
    else:
        residues = sorted(data1)

    num_residues = len(residues)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2.5), constrained_layout=True)
    axs = axs.flatten()

    for i, res in enumerate(residues):
        ax = axs[i]

        # Plot Dataset 1
        if res in data1:
            pts1 = sorted(data1[res])
            ets1 = sorted(error1[res])

            n1, i_over_i0_1 = zip(*pts1)
            _, rel_errors1 = zip(*ets1)

            T2s1, T2errors1 = calc_T2_per_point(i_over_i0_1, tau, rel_errors1)

            filtered = [(n, t, e) for n, t, e in zip(n1, T2s1, T2errors1) if tmax > t > tmin]

            if filtered:
                n1, T2s1, T2errors1 = zip(*filtered)
                ax.errorbar(n1, T2s1, yerr=T2errors1, fmt='-o', color='tab:blue', capsize=1.5, markersize=4)

        # Plot Dataset 2
        if data2 and res in data2:
            pts2 = sorted(data2[res])
            errors2 = sorted(error2[res])

            n2, i_over_i0_2 = zip(*pts2)
            _, rel_errors2 = zip(*errors2)

            T2s2, T2errors2 = calc_T2_per_point(i_over_i0_2, tau, rel_errors2)

            filtered = [(n, t, e) for n, t, e in zip(n2, T2s2, T2errors2) if tmax > t > tmin]

            if filtered:
                n2, T2s2, T2errors2 = zip(*filtered)
                ax.errorbar(n2, T2s2, yerr=T2errors2, fmt='-o', color='tab:red', capsize=1.5, markersize=4)

        # Labels and title
        name = name_map1.get(res) or (name_map2.get(res) if name_map2 else f"Res {res}")
        ax.set_title(f'{name} {res}', fontsize=8)
        ax.set_xlabel('Echo Number (n)', fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel(r'$T_2^{\mathrm{eff}}$ (ms)', fontsize=7)

        # Limits
        ax.set_xlim(-5, 65)
        ax.set_xticks(np.linspace(0, 60, 3))

        # Y limits (auto)
        all_y = []
        if res in data1:
            all_y.extend(T2s1)
        if data2 and res in data2:
            all_y.extend(T2s2)

        all_y = [y for y in all_y if np.isfinite(y)]
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            ypad = 3 * (ymax - ymin) if ymax != ymin else 0.05
            ax.set_ylim(ymin - ypad, ymax + ypad)
        else:
            ax.set_ylim(0, 50)  # fallback

    # Hide unused subplots
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.suptitle(
        fr"CPMG $T_2^{{\mathrm{{eff}}}}$ per Residue (fixed τ = {tau} ms, filtered between {tmin}–{tmax} ms)",
        fontsize=14
    )
    plt.show()

def calc_R2_per_point(T2_eff, relative_uncertainty):
    """Given a list of T2eff values and relative uncertainties, return the corresponding R2 values (s-1) and errors."""
    R2_eff = (1 / np.array(T2_eff)) * 1000
    R2_errors = R2_eff * np.array(relative_uncertainty)

    return R2_eff, R2_errors

def plot_R2_per_point(tau, data1, error1, name_map1, data2=None, error2=None, name_map2=None):
    """Plot R2 per point vs echo number (n) for all residues, using fixed total relaxation delay."""

    rmin = 0
    rmax = 100

    if data2:
        residues = sorted(set(data1) & set(data2))
    else:
        residues = sorted(data1)

    num_residues = len(residues)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2.5), constrained_layout=True)
    axs = axs.flatten()

    for i, res in enumerate(residues):
        ax = axs[i]

        # Plot Dataset 1
        if res in data1:
            pts1 = sorted(data1[res])
            ets1 = sorted(error1[res])
            n1, i_over_i0_1 = zip(*pts1)
            _, rel_errors1 = zip(*ets1)
            T2s1, _ = calc_T2_per_point(i_over_i0_1, tau, rel_errors1)
            R2s1, R2errors1 = calc_R2_per_point(T2s1, rel_errors1)
            filtered = [(n, r, e) for n, r, e in zip(n1, R2s1, R2errors1) if rmax > r > rmin]
            if filtered:
                n1, R2s1, R2errors1 = zip(*filtered)
                ax.errorbar(n1, R2s1, R2errors1, fmt='-o', color='tab:blue', capsize=1.5, markersize=4)

        # Plot Dataset 2
        if data2 and res in data2:
            pts2 = sorted(data2[res])
            ets2 = sorted(error2[res])
            n2, i_over_i0_2 = zip(*pts2)
            _, rel_errors2 = zip(*ets2)
            T2s2, _ = calc_T2_per_point(i_over_i0_2, tau, rel_errors2)
            R2s2, R2errors2 = calc_R2_per_point(T2s2, rel_errors2)
            filtered = [(n, r, e) for n, r, e in zip(n2, R2s2, R2errors2) if rmax > r > rmin]
            if filtered:
                n2, R2s2, R2errors2 = zip(*filtered)
                ax.errorbar(n2, R2s2, R2errors2, fmt='-o', color='tab:red', capsize=1.5, markersize=4)

        # Labels and title
        name = name_map1.get(res) or (name_map2.get(res) if name_map2 else f"Res {res}")
        ax.set_title(f'{name} {res}', fontsize=8)
        ax.set_xlabel('Echo Number (n)', fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel(r'$R_2^{\mathrm{eff}}$ (s$^{-1}$)', fontsize=7)

        # Limits
        ax.set_xlim(-5, 65)
        ax.set_xticks(np.linspace(0, 60, 3))

        # Y limits (auto)
        all_y = []
        if res in data1:
            all_y.extend(R2s1)
        if data2 and res in data2:
            all_y.extend(R2s2)

        all_y = [y for y in all_y if np.isfinite(y)]
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            ypad = 3 * (ymax - ymin) if ymax != ymin else 0.05
            ax.set_ylim(ymin - ypad, ymax + ypad)
        else:
            ax.set_ylim(0, 50)  # fallback

    # Hide unused subplots
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.suptitle(
        fr"CPMG $R_2^{{\mathrm{{eff}}}}$ per Residue (fixed τ = {tau} ms)", # filtered between ${tmin}\!-\!{tmax}\,\mathrm{{s}}^{{-1}}$)",
        fontsize=14
    )
    plt.show()

# --- Main Execution ---
def main():
    configure_matplotlib()
    args = parse_args()

    data1, apo1, _, scans1, _, baseline_noise_1 = read_data(args.file1)
    tau = extract_tau(data1)
    id_to_name_mapping1 = {entry['res']: entry['name'] for entry in data1 if entry['res'] is not None}
    apo1 = args.apo_spectrum if args.apo_spectrum else apo1

    data_spec1 = organise_by_spectrum(data1, value_key='height')
    comparisons1, noise_comparisons1, relative_uncertainty_comparisons_1 = calculate_height(data_spec1, apo1, scans1, baseline_noise_1)
    res_intensity_data1, res_error_data1, res_relative_uncertainty_data1 = collect_intensity_data(data1, comparisons1, noise_comparisons1, relative_uncertainty_comparisons_1)

    res_intensity_data2 = None
    res_error_data2 = None
    id_to_name_mapping2 = None

    if args.file2:
        data2, apo2, _, scans2, _, baseline_noise_2 = read_data(args.file2)
        id_to_name_mapping2 = {entry['res']: entry['name'] for entry in data2 if entry['res'] is not None}
        data_spec2 = organise_by_spectrum(data2, value_key='height')
        comparisons2, noise_comparisons2, relative_uncertainty_comparisons_2 = calculate_height(data_spec2, apo2, scans2, baseline_noise_2)
        res_intensity_data2, res_error_data2, res_relative_uncertainty_data2 = collect_intensity_data(data2, comparisons2, noise_comparisons2, relative_uncertainty_comparisons_2)

    plot_cpmg_intensity_ratios(
        tau,
        res_intensity_data1, 
        res_error_data1,
        id_to_name_mapping1, 
        res_intensity_data2, 
        res_error_data2,
        id_to_name_mapping2
    )

    plot_T2_per_point(
        tau, 
        res_intensity_data1, 
        res_relative_uncertainty_data1,
        id_to_name_mapping1, 
        res_intensity_data2, 
        res_relative_uncertainty_data2,
        id_to_name_mapping2
    )

    plot_R2_per_point(
        tau, 
        res_intensity_data1, 
        res_relative_uncertainty_data1,
        id_to_name_mapping1, 
        res_intensity_data2, 
        res_relative_uncertainty_data2,
        id_to_name_mapping2
    )

if __name__ == "__main__":
    main()
