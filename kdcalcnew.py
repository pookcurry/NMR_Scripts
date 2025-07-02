# Example command line input
# python kdcalc.py data.csv -ht 0.04 -nt 0.12 -kd 100 -pl -pg -bs

import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit, least_squares
from functools import partial
from tqdm import trange
import argparse
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
    parser = argparse.ArgumentParser(description="Fit CSP data to binding model.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("-ht", "--h_threshold", type=float, default=0.03, help="Minimum threshold to consider H shifts. Default is 0.03 ppm")
    parser.add_argument("-nt", "--n_threshold", type=float, default=0.15, help="Minimum threshold to consider N shifts. Default is 0.15 ppm")
    parser.add_argument("-kd", "--kd_threshold", type=float, default=200, help="Maximum Kd for filtering residue fits. Default is 200 µM")
    parser.add_argument("-pl", "--plotlocal", action="store_true", help="Plot fitted local Kd curves per residue")
    parser.add_argument("-pg", "--plotglobal", action="store_true", help="Plot fitted global Kd curves per residue")
    parser.add_argument("-bs", "--bootstrap", action="store_true", help="Enable bootstrapping for error estimation")
    return parser.parse_args()

# --- Data Preparation Functions ---
def read_data(filepath):
    """Read titration CSV and return structured list of entries."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.iloc[:, [3, 4, 7, 8, 12, 13, 14, 17]]
    df.columns = ['series_step_x', 'additional_series_step_x', 'ppm', 'height', 'res', 'name', 'atom', 'spectrum']

    df['series_step_x'] = pd.to_numeric(df['series_step_x'], errors='coerce')
    df['additional_series_step_x'] = pd.to_numeric(df['additional_series_step_x'], errors='coerce')
    df['ppm'] = pd.to_numeric(df['ppm'], errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['res'] = pd.to_numeric(df['res'], errors='coerce')

    df = df[df['name'].notna()]

    apo_row = df[df['series_step_x'] == 0].head(1)
    apo_spectrum = apo_row['spectrum'].iloc[0] if not apo_row.empty else None

    return df.to_dict(orient='records'), apo_spectrum

def organize_by_spectrum(data):
    """Organize shift data by spectrum -> residue -> atom."""
    data_spec = defaultdict(lambda: defaultdict(dict))
    for entry in data:
        if entry['spectrum']:
            data_spec[entry['spectrum']][entry['res']][entry['atom']] = entry['ppm']
    return data_spec

def extract_protein_concentration(data: list[dict]) -> float | None:
    """Get the protein concentration T from metadata."""
    T_values = {entry['additional_series_step_x'] for entry in data if entry['additional_series_step_x'] is not None}
    if len(T_values) == 1:
        return T_values.pop()
    print("\n⚠️ Multiple T values found; cannot determine global T. Need to use another binding model (not implemented yet).")
    return None

# --- CSP and Fitting Functions ---
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

def collect_csp_data(data, csps_per_comparison):
    """Collect ligand concentration and CSPs for each residue."""
    residue_csp_data = defaultdict(list)
    data_index = {(entry['spectrum'], entry['res']): entry for entry in data if entry['series_step_x'] is not None}
    for spectrum, csps in csps_per_comparison.items():
        for result in csps:
            res = result['Residue']
            csp = result['CSP']
            entry = data_index.get((spectrum, res))
            if entry:
                residue_csp_data[res].append((entry['series_step_x'], csp))
    return residue_csp_data

def quadratic_binding_model(x, Bmax, Kd, T):
    """Quadratic binding model."""
    return Bmax * ((T + x + Kd - np.sqrt((T + x + Kd)**2 - 4 * T * x)) / (2 * T))

def fit_residue_binding(csp_data, T):
    """Fit Kd per residue using quadratic binding model."""
    model = partial(quadratic_binding_model, T=T)
    results = {}
    for res, points in csp_data.items():
        if len(points) < 3:
            continue
        concentrations, csps = zip(*sorted(points))
        concentrations = [0.0] + list(concentrations)
        csps = [0.0] + list(csps)
        try:
            popt, _ = curve_fit(model, concentrations, csps, p0=[max(csps), 100.0],
                                bounds=([0, 1e-6], [np.inf, np.inf]))
            results[res] = popt[1]
        except RuntimeError:
            print(f"❌ Fit failed for residue {res}")
    return results

def filter_by_csp_shift(data_spec, apo, h_thresh, n_thresh):
    """Filter residues by significant CSP shift from apo spectrum."""
    valid = set()
    for spec in data_spec:
        if spec == apo:
            continue
        for res in data_spec[apo]:
            if res not in data_spec[spec]:
                continue
            h_apo = data_spec[apo][res].get('H')
            h_other = data_spec[spec][res].get('H')
            n_apo = data_spec[apo][res].get('N')
            n_other = data_spec[spec][res].get('N')
            if h_apo and h_other and abs(h_apo - h_other) >= h_thresh:
                valid.add(res)
            if n_apo and n_other and abs(n_apo - n_other) >= n_thresh:
                valid.add(res)
    return valid

def prepare_global_fit_data(filtered_csp_data):
    """Prepare data arrays for global fitting."""
    global_concs, global_csps, residue_ids = [], [], []
    for res, datapoints in filtered_csp_data.items():
        for conc, csp in datapoints:
            global_concs.append(conc)
            global_csps.append(csp)
            residue_ids.append(res)
    return global_concs, global_csps, residue_ids

def global_binding_model(params, concs, residues, T, residue_to_index):
    """Model used for global fit."""
    Kd = params[0]
    predictions = []
    for x, r in zip(concs, residues):
        Bmax = params[1 + residue_to_index[r]]
        term = (T + x + Kd - np.sqrt((T + x + Kd)**2 - 4 * T * x)) / (2 * T)
        predictions.append(Bmax * term)
    return np.array(predictions)

def fit_global_kd(filtered_csp_data, T, residue_to_index):
    unique_residues = list(filtered_csp_data)
    residue_to_index = {res: i for i, res in enumerate(unique_residues)}
    global_concs, global_csps, residue_ids = prepare_global_fit_data(filtered_csp_data)
    initial_guess = [10.0] + [max([csp for _, csp in filtered_csp_data[res]]) for res in unique_residues]
    result = least_squares(
        lambda p, x, y, r: global_binding_model(p, x, r, T, residue_to_index) - y,
        x0=initial_guess,
        args=(np.array(global_concs), np.array(global_csps), residue_ids),
        bounds=([1e-6] + [0] * len(unique_residues), [np.inf] * (1 + len(unique_residues)))
    )
    # Estimate covariance matrix of the fit
    residuals = result.fun
    J = result.jac
    _, s, VT = np.linalg.svd(J, full_matrices=False)
    threshold = np.finfo(float).eps * max(J.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:len(s)]
    cov = VT.T @ np.diag(1 / s**2) @ VT
    residual_variance = np.sum(residuals**2) / (len(global_concs) - len(result.x))
    cov *= residual_variance
    kd_global_se = np.sqrt(cov[0, 0])
    return result.x[0], kd_global_se, result, residue_to_index

def print_summary(kd_results_clean, id_to_name_mapping, kd_thresh, h_thresh, n_thresh, kd_global, kd_global_se):
    print(f"\n📊 Estimated Kd values per filtered residue:"
        f"\n(Filtered by Kd < {kd_thresh} µM and Δ1H ≥ {h_thresh} ppm or Δ15N ≥ {n_thresh} ppm)")
    print(f"\n📈 Mean Kd (filtered) = {np.mean(list(kd_results_clean.values())):.2f} µM ± {np.std(list(kd_results_clean.values()), ddof=1):.2f} µM (1σ)")
    for res, kd in sorted(kd_results_clean.items()):
        print(f"{id_to_name_mapping[res]} {int(res)}: Kd = {kd:.2f} µM")
    print(f"\n🌍 Global Kd (fitted) = {kd_global:.2f} µM ± {kd_global_se:.2f} µM (SE)")
    print(", ".join(f"{id_to_name_mapping[res]}{int(res)}" for res in sorted(kd_results_clean.keys())), "\n")

def plot_local_fits(csp_data, kd_results, T, id_to_name_mapping):
    valid_items = [(res, csp_data[res]) for res in csp_data if int(res) in kd_results]
    num_residues = len(valid_items)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), constrained_layout=True)
    axs = axs.flatten()

    max_conc = max(max(c[0] for c in csp_data[res]) for res in kd_results)
    x_fit = np.linspace(0, max_conc * 1.1, 100)

    sorted_items = sorted(valid_items, key=lambda item: int(item[0]))

    for i, (res, points) in enumerate(sorted_items):
        ax = axs[i]
        points = sorted(points)
        concs, csps = zip(*points)
        res_int = int(res)  # Normalize key
        if res_int not in kd_results:
            ax.axis('off')
            continue
        points = sorted(csp_data[res])
        
        model = partial(quadratic_binding_model, T=T)
        Bmax = max(csps)
        Kd = kd_results[res]
        y_fit = model(x_fit, Bmax, Kd)

        ax.plot(concs, csps, 'o')  # Don't add label to data points
        fit_line, = ax.plot(x_fit, y_fit, '-', label=f'Kd = {Kd:.2f} µM')  # Capture the fit line        
        label = f"{id_to_name_mapping.get(res, 'Res')} {int(res)}"
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Ligand (µM)")
        tick_num = 4  # Or 8 or 10, depending on how dense you want it
        ax.set_xticks(np.linspace(0, max(concs) * 1.0, tick_num))
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend([fit_line], [f'Kd = {Kd:.2f} µM'], loc='lower center', ncol=1, fontsize=6)

        if (i % ncols) == 0:
            ax.set_ylabel("CSP (ppm)")
        else:
            ax.set_ylabel("")

    # Hide unused axes if any
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.legend(
        ["CSP Points"], 
        loc='lower center', ncol=2, fontsize=10, 
        # bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure
        )

    fig.suptitle("Local Kd Fits per Residue", fontsize=16)
    plt.show()

def plot_global_fits(filtered_csp_data, kd_global, result, T, residue_to_index, id_to_name_mapping):
    num_residues = len(filtered_csp_data)
    ncols = 10
    nrows = math.ceil(num_residues / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), constrained_layout=True)
    axs = axs.flatten()

    x_fit = np.linspace(0, max([c for res in filtered_csp_data for c, _ in filtered_csp_data[res]]) * 1.1, 100)

    sorted_items = sorted(filtered_csp_data.items(), key=lambda item: int(item[0]))

    for i, (res, points) in enumerate(sorted_items):
        ax = axs[i]

        # Ensure points is a list of tuples
        points = sorted(points)
        concs, csps = zip(*points)
        Bmax = result.x[1 + residue_to_index[res]]
        y_fit = quadratic_binding_model(x_fit, Bmax, kd_global, T)

        ax.plot(concs, csps, 'o', label='Data')
        ax.plot(x_fit, y_fit, '-', label=f'Kd = {kd_global:.2f} µM')

        label = f"{id_to_name_mapping.get(res, 'Res')} {int(res)}"
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Ligand (µM)")
        tick_num = 4  # Or 8 or 10, depending on how dense you want it
        ax.set_xticks(np.linspace(0, max(concs) * 1.0, tick_num))
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        if (i % ncols) == 0:
            ax.set_ylabel("CSP (ppm)")
        else:
            ax.set_ylabel("")

        ax.legend().set_visible(False)  # Hide individual legends

    # Shared legend at bottom
    fig.legend(
        ["CSP Points", f"Globally Fitted Kd = {kd_global:.2f} µM"], 
        loc='lower center', ncol=2, fontsize=10, 
        # bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure
    )

    # Hide unused subplots
    for ax in axs[num_residues:]:
        ax.axis('off')

    fig.suptitle("Global Kd Fit per Residue", fontsize=16)
    plt.show()

def bootstrap(filtered_csp_data, T, kd_global, result, residue_to_index):
    unique_residues = list(filtered_csp_data)
    global_concs, global_csps, residue_ids = prepare_global_fit_data(filtered_csp_data)
    initial_guess = [kd_global] + result.x[1:].tolist()
    n_iterations = 500
    kd_bootstrap = []
    for i in trange(n_iterations, desc="🔄 Bootstrapping"):
        indices = np.random.choice(len(global_concs), len(global_concs), replace=True)
        boot_concs = np.array(global_concs)[indices]
        boot_csps = np.array(global_csps)[indices]
        boot_residues = np.array(residue_ids)[indices]

        try:
            boot_result = least_squares(
                lambda p, x, y, r: global_binding_model(p, x, r, T, residue_to_index) - y,
                x0=initial_guess,
                args=(boot_concs, boot_csps, boot_residues),
                bounds=([1e-6] + [0] * len(unique_residues), [np.inf] * (1 + len(unique_residues)))
            )
            kd_bootstrap.append(boot_result.x[0])
        except Exception as e:
            warnings.warn(f"Bootstrap iteration failed: {e}")
            continue

    # Compute statistics
    mean_kd = np.mean(kd_bootstrap)
    std_kd = np.std(kd_bootstrap, ddof=1)
    ci_lower = np.percentile(kd_bootstrap, 2.5)
    ci_upper = np.percentile(kd_bootstrap, 97.5)

    # Summary of bootstrap results
    print(f"Bootstrapping completed over {len(kd_bootstrap)} successful iterations.")
    print(f"\n🌍 Global Kd (bootstrapped) = {mean_kd:.2f} µM ± {std_kd:.2f} µM (95% CI: {ci_lower:.2f} – {ci_upper:.2f} µM)\n")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(kd_bootstrap, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(mean_kd, color='red', linestyle='--', label=f"Mean = {mean_kd:.2f} µM")
    plt.axvline(ci_lower, color='green', linestyle=':', label=f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}] µM")
    plt.axvline(ci_upper, color='green', linestyle=':')
    plt.title("Bootstrapped Global Kd Estimates")
    plt.xlabel("Kd (µM)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    configure_matplotlib()
    args = parse_args()
    data, apo_spectrum = read_data(args.input_file)
    id_to_name_mapping = {entry['res']: entry['name'] for entry in data if entry['res'] is not None}
    data_spec = organize_by_spectrum(data)
    T = extract_protein_concentration(data)
    if T is None:
        raise ValueError("Protein concentration T could not be determined from metadata.")
    
    csps_per_comparison = calculate_csp(data_spec, apo_spectrum)
    residue_csp_data = collect_csp_data(data, csps_per_comparison)

    kd_results = fit_residue_binding(residue_csp_data, T)
    valid_residues = filter_by_csp_shift(data_spec, apo_spectrum, args.h_threshold, args.n_threshold)

    kd_results_clean = {
        res: kd for res, kd in kd_results.items() 
        if 0 < kd < args.kd_threshold and res in valid_residues
    }
    
    filtered_csp_data = {
        res: residue_csp_data[res] 
        for res in kd_results_clean
    }

    residue_to_index = {res: i for i, res in enumerate(filtered_csp_data)}

    kd_global, kd_global_se, result, residue_to_index = fit_global_kd(
        filtered_csp_data, T, residue_to_index
    )

    print_summary(
        kd_results_clean, 
        id_to_name_mapping, 
        args.kd_threshold, 
        args.h_threshold, 
        args.n_threshold, 
        kd_global, 
        kd_global_se,
    )

    if args.plotlocal:
        print("🖼️ Plotting local Kd fits...\n")
        plot_local_fits(residue_csp_data, kd_results_clean, T, id_to_name_mapping)
    else:
        print("⏩ Skipping local Kd plotting. \nUse '-pl' or '--plotlocal' to enable.\n")

    if args.plotglobal:
        print("🖼️ Plotting global Kd fits...\n")
        plot_global_fits(filtered_csp_data, kd_global, result, T, residue_to_index, id_to_name_mapping)
    else:
        print("⏩ Skipping global Kd plotting. \nUse '-pg' or '--plotglobal' to enable.\n")

    if args.bootstrap:
        bootstrap(filtered_csp_data, T, kd_global, result, residue_to_index)
    else:
        print("⏩ Skipping bootstrapping. \nUse '-bs' or '--bootstrap' to enable.\n")

if __name__ == "__main__":
    main()
