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
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("-seq", "--sequence_file", type=str, help="1-letter amino acid sequence (e.g., matloop.txt)")
    parser.add_argument("-psi", "--psipred_file", help="PSIPRED .ss2 file for secondary structure overlay")
    parser.add_argument("-xlim", "--x_limit", type=float, default=None, help="Override the X-axis limit (residue number)")
    parser.add_argument("-ylim", "--y_limit", type=float, default=None, help="Override the Y-axis limit (CSP or IL Ratio)")
    parser.add_argument("-il", "--intensity_loss", action="store_true", help="Enable to plot intensity losses instead of CSPs")
    parser.add_argument("-apo", "--apo_spectrum", type=str, help="Explicitly specify the name of the apo spectrum \n(could be useful if there are issues with intensity)")
    parser.add_argument("-s", "--save", action="store_true", help="Enable to save plots as .svg files")
    parser.add_argument("-skip", "--skip_nterm", type=int, default=0, help='Number of N-terminal residues to skip in plots (default: 0)')
    parser.add_argument("-ilratio", "--intensitylossratio", action="store_true", help="Enable to plot the average intensity ratios per spectra")
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

def calculate_csp(data_spec, apo):
    """Calculate CSPs vs apo spectrum for each residue."""
    comparisons = {}
    spectra = set(data_spec) - {apo}

    for spec in spectra:
        csps = []
        for res in data_spec[apo]:
            if res > 0 and res in data_spec[spec]:
                h_apo, n_apo = data_spec[apo][res].get('H'), data_spec[apo][res].get('N')
                h_other, n_other = data_spec[spec][res].get('H'), data_spec[spec][res].get('N')
                if all([h_apo, n_apo, h_other, n_other]):
                    delta_h = h_apo - h_other
                    delta_n = n_apo - n_other
                    csp = np.sqrt(delta_h**2 + (0.2 * delta_n)**2)
                    csps.append({'Residue': res, 'CSP': csp})
        comparisons[spec] = csps
    return comparisons

def calculate_height(data_spec, apo, ns_dict):
    comparisons = {}
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

        for res in data_spec[apo]:
            if res > 0 and res in data_spec[spec]:
                h_apo = data_spec[apo][res].get('H_scaled')
                h_other = data_spec[spec][res].get('H_scaled')

                if h_apo and h_other:
                    ratio = h_other / h_apo if h_apo != 0 else 0
                    heights.append({'Residue': res, 'Ratio': ratio})

        comparisons[spec] = heights

    return comparisons

def average_ratios(comparisons, data_spec, apo):
    all_residues = sorted(data_spec[apo].keys())  # assumes apo has all relevant residues
    averaged = {}

    for spec, results in comparisons.items():
        ratio_dict = {entry['Residue']: entry['Ratio'] for entry in results}
        ratios = [ratio_dict.get(res, 0) for res in all_residues]
        avg = sum(ratios) / len(ratios) if ratios else 0
        averaged[spec] = avg

    return averaged

def plot_average_ratios(avg_ratios, molar_ratios, apo=None, save=False, filename="average_intensities.svg"):
    spectra = sorted(avg_ratios.keys(), key=lambda s: molar_ratios.get(s, float('inf')))
    x_vals = [molar_ratios.get(spec, None) for spec in spectra]
    y_vals = [avg_ratios[spec] for spec in spectra]

    print(x_vals)
    print(y_vals)

    # Filter out any None values in molar_ratios
    x_vals, y_vals = zip(*[(x, y) for x, y in zip(x_vals, y_vals) if x is not None])

    plt.figure(figsize=(8, 5))
    plt.scatter([0], [1], color='dodgerblue', s=80)
    plt.scatter(x_vals, y_vals, color='dodgerblue', s=80)
    plt.xlabel("Molar Ratio (Protein:Substrate)")
    plt.ylabel("Average Intensity Ratio")
    plt.title("Average Intensity Ratios vs Molar Ratio")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(filename)
    else:
        plt.show()

def find_unassigned_residues_in_apo(residues_present, sequence):
    """Return list of residues unassigned in apo compared to expected NMR residue numbers."""
    if not sequence:
        return []
    expected_residues = set(range(1, len(sequence) + 1))
    unassigned = sorted(expected_residues - set(residues_present))
    return unassigned

def find_missing_residues_in_spectra(data_spec, apo):
    """Return list of residues that disappear during each titration compared to apo."""
    missing_by_spec = {}
    apo_residues = set(data_spec[apo].keys())

    for spectrum, res_dict in data_spec.items():
        if spectrum == apo:
            continue
        res_present = set(res_dict.keys())
        missing = sorted(apo_residues - res_present)
        missing_by_spec[spectrum] = missing

    return missing_by_spec

def plot_csp(data_spec, apo, residues_present, sequence=None, psipred_file=None, y_limit=None, x_limit=None, save=False, skip_nterm=0):
    csps_per_comparison = calculate_csp(data_spec, apo)

    # --- Get global max CSP and residue number ---
    all_csps = [result['CSP'] for csps in csps_per_comparison.values() for result in csps]
    all_residues = [result['Residue'] for csps in csps_per_comparison.values() for result in csps]

    global_max_csp = max(all_csps)
    global_max_res = max(all_residues)

    # Round Y-limit up to nearest 0.05 and clamp to max 0.5
    ymax = y_limit if y_limit is not None else min(np.ceil((global_max_csp + 1e-6) / 0.05) * 0.05, 0.5)
    xmin = skip_nterm
    xmax = x_limit if x_limit else len(sequence) + 1 if sequence else global_max_res + 1

    for spectrum, csps in csps_per_comparison.items():
        res = [result['Residue'] for result in csps]
        csp = [result['CSP'] for result in csps]

        # Calculate standard deviation
        stdev = np.std(csp)

        # Assign colors based on threshold
        col = ['#9da3a4' if val <= stdev else '#db7f8e' for val in csp]
        
        # Plotting
        plt.figure(figsize=(7, 3.5))
        ax = plt.subplot(111)
    
        # unassigned = find_unassigned_residues_in_apo(residues_present, sequence)
        # if unassigned:
        #     ax.bar(
        #         unassigned,
        #         [ymax] * len(unassigned),
        #         width=1,
        #         color='#cbe9f8',
        #         edgecolor='none',
        #         zorder=0
        #     )

        if sequence:
            map_sequence(ax, residues_present, sequence, skip_nterm)
        else:
            ax.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True)

        if psipred_file:
            plot_secondary_structure(ax, psipred_file, skip_nterm)

        ax.bar(res, csp, 1, color=col, edgecolor='k')
        tick_positions = np.arange(xmin, xmax, 10)
        ax.set_xticks(tick_positions)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x - skip_nterm)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        plt.xlabel('Residue')
        plt.ylabel(r'{hydrogen} + {nitrogen} CSP (ppm)'.format(hydrogen=r'$^{1}$H', nitrogen=r'$^{15}$N$_{\mathrm{H}}$'))
        plt.title(f'{apo} vs {spectrum}', y=1.16 if sequence and psipred_file else 1.08 if sequence else 1)
        plt.ylim(0, ymax)
        plt.xlim(xmin, xmax)
        plt.axhline(y=stdev, color='black', linestyle='--', linewidth='1')
        plt.tight_layout()

        missing_all_spectra = find_missing_residues_in_spectra(data_spec, apo) # returns a dictionary
        missing_per_spectrum = missing_all_spectra.get(spectrum, []) # returns a list per spectrum
        if missing_per_spectrum and sequence:
            # bottom = ax.get_ylim()[0]
            for res in missing_per_spectrum:
                # ax.text(
                #     res,
                #     bottom,
                #     '*',
                #     color='black',
                #     ha='center',
                #     va='bottom',
                #     fontsize=get_optimal_font_size(ax, sequence),
                #     fontweight='bold',
                #     zorder=5
                # )

                ax.bar(
                    res,
                    [ymax] * len(missing_per_spectrum),
                    width=1,
                    color="#cbe9f8",
                    edgecolor='none',
                    zorder=0
                )

        if save:
            name = sanitise_filename(f"{apo}_vs_{spectrum}")
            plt.savefig(f"{name}_CSP.svg")
        else:
            plt.show()

def plot_intensity(data_spec, ns_dict, apo, residues_present, sequence=None, psipred_file=None, y_limit=None, x_limit=None, save=False, skip_nterm=0):
    heights_per_comparison = calculate_height(data_spec, apo, ns_dict)

    # --- Get global max height and residue number ---
    all_ratios = [result['Ratio'] for heights in heights_per_comparison.values() for result in heights]
    all_residues = [result['Residue'] for heights in heights_per_comparison.values() for result in heights]

    global_max_height = max(all_ratios)
    global_max_res = max(all_residues)

    # Round Y-limit up to nearest 0.2 and clamp to max 10
    ymax = y_limit if y_limit is not None else min(np.ceil((global_max_height + 1e-6) / 0.2) * 0.2, 10)
    xmin = skip_nterm
    xmax = x_limit if x_limit else len(sequence) + 1 if sequence else global_max_res + 1

    for spectrum, heights in heights_per_comparison.items():
        res = [result['Residue'] for result in heights]
        height = [result['Ratio'] for result in heights]

        # Plotting
        plt.figure(figsize=(7, 3.5))
        ax = plt.subplot(111)

        # unassigned = find_unassigned_residues_in_apo(residues_present, sequence)
        # if unassigned:
        #     ax.bar(
        #         unassigned,
        #         [ymax] * len(unassigned),
        #         width=1.1,
        #         color='#cbe9f8',
        #         edgecolor='none',
        #         zorder=0
        #     )

        if sequence:
            map_sequence(ax, residues_present, sequence, skip_nterm)
        else:
            ax.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True)

        if psipred_file:
            plot_secondary_structure(ax, psipred_file, skip_nterm)

        ax.bar(res, height, 1, color='#48ad71', edgecolor='k')
        tick_positions = np.arange(xmin, xmax, 10)
        ax.set_xticks(tick_positions)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x - skip_nterm)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        plt.xlabel('Residue')
        plt.ylabel(r'Intensity ratio ($I_{\mathrm{bound}}$ / $I_{\mathrm{free}}$)')
        plt.title(f'{apo} vs {spectrum}', y=1.16 if sequence and psipred_file else 1.08 if sequence else 1)
        plt.ylim(0, ymax)
        plt.xlim(xmin, xmax)
        plt.axhline(y=1, color='black', linestyle='--', linewidth='1')
        plt.tight_layout()

        missing_all_spectra = find_missing_residues_in_spectra(data_spec, apo) # returns a dictionary
        missing_per_spectrum = missing_all_spectra.get(spectrum, []) # returns a list per spectrum
        if missing_per_spectrum and sequence:
            # bottom = ax.get_ylim()[0]
            for res in missing_per_spectrum:
                # ax.text(
                #     res,
                #     bottom,
                #     '*',
                #     color='black',
                #     ha='center',
                #     va='bottom',
                #     fontsize=get_optimal_font_size(ax, sequence),
                #     fontweight='bold',
                #     zorder=5
                # )

                ax.bar(
                    res,
                    [ymax] * len(missing_per_spectrum),
                    width=1,
                    color="#cbe9f8",
                    edgecolor='none',
                    zorder=0
                )


        if save:
            name = sanitise_filename(f"{apo}_vs_{spectrum}")
            plt.savefig(f"{name}_intensity.svg")
        else:
            plt.show()

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

# --- Main Execution ---
def main():
    configure_matplotlib()
    args = parse_args()
    sequence = read_sequence(args.sequence_file) if args.sequence_file else None
    if not sequence:
        print("\n⏩ Skipping sequence plotting. \nUse '-seq' or '--sequence_file' with a .txt of the sequence to overlay above the plot.")
    if not args.psipred_file:
        print("\n⏩ Skipping secondary structure plotting. \nUse '-psi' or '--psipred_file' with a .ss2 from PSIPRED to overlay above the plot.")

    data, apo, residues_present, scans, molar_ratio = read_data(args.input_file)
    apo = args.apo_spectrum if args.apo_spectrum else apo
    
    if args.intensity_loss or args.intensitylossratio:
        data_spec = organise_by_spectrum(data, value_key='height')
        comparisons = calculate_height(data_spec, apo, scans)

        if args.intensitylossratio:
            avg_ratios = average_ratios(comparisons, data_spec, apo)
            plot_average_ratios(avg_ratios, molar_ratio, apo, save=args.save)

        else:
            plot_intensity(
                data_spec,
                scans,
                apo=apo,
                residues_present=residues_present,
                sequence=sequence,
                psipred_file=args.psipred_file,
                y_limit=args.y_limit,
                x_limit=args.x_limit,
                save=args.save,
                skip_nterm=args.skip_nterm
            )

    else:
        data_spec = organise_by_spectrum(data, value_key = 'ppm')
        plot_csp(
            data_spec,
            apo=apo,
            residues_present=residues_present,
            sequence=sequence,
            psipred_file=args.psipred_file,
            y_limit=args.y_limit,
            x_limit=args.x_limit,
            save=args.save,
            skip_nterm=args.skip_nterm
        )

if __name__ == "__main__":
    main()
