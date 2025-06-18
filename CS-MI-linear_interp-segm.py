# -*- coding: utf-8 -*-
"""
Unified script for time series analysis of the economic and energy sector.

This script performs two main analyses on a dataset:
1.  Rolling window Spearman's Correlation.
2.  Rolling window Mutual Information.

Data is read from local directories, eliminating the need for external APIs or databases.
The outputs are heatmaps saved as .png images in their respective output directories.

Author: [Your Name/Your Organization]
Date: June 18, 2025
"""

# ==============================================================================
# 1. LIBRARY IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path  # For cross-platform path handling
import datetime
from parallel_pandas import ParallelPandas

# Initialize parallel_pandas
ParallelPandas.initialize(n_cpu=4, split_factor=4, disable_pr_bar=False)

# ==============================================================================
# 2. GLOBAL CONFIGURATIONS
# ==============================================================================
# --- Analysis Period ---
# Define the start and end dates to filter the data.
START_DATE = datetime.datetime(2014, 12, 16)
END_DATE = datetime.datetime(2023, 4, 28)

# --- Input Directories ---
# Define the names of the folders where the input data files are located.
# These directories must exist in the same folder as this script.
MARKET_DATA_DIR = Path("./linear_interp")
API_DATA_DIR = Path("./dados")

# --- Output Directories ---
# Define the names of the folders where the generated images will be saved.
# These folders will be created automatically if they do not exist.
SPEARMAN_OUTPUT_DIR = Path("./CS-linear-interp")
MI_OUTPUT_DIR = Path("./MI-linear-interp")

# --- Plotting Configurations ---
plt.style.use("seaborn-whitegrid")
sns.set(font_scale=2.4)
SPEARMAN_FIGSIZE = (30, 20)
MI_FIGSIZE = (30, 10)
SAVE_DPI = 400

# ==============================================================================
# 3. DATA LOADING AND PREPROCESSING FUNCTIONS
# ==============================================================================

def load_market_data(directory_path, start_date, end_date):
    """
    Loads and consolidates all CSV files from the market data directory.
    
    Args:
        directory_path (Path): Path object for the 'linear_interp' directory.
        start_date (datetime): Start date for filtering.
        end_date (datetime): End date for filtering.

    Returns:
        pd.DataFrame: DataFrame containing all consolidated market data.
    """
    df_list = []
    print(f"🔎 Loading market data from directory: {directory_path}")
    for file_path in directory_path.glob("*.csv"):
        try:
            column_name = file_path.stem  # Use the filename without extension as the column name
            temp_df = pd.read_csv(file_path)
            
            # Rename value and date columns if necessary
            if "result" in temp_df.columns:
                temp_df = temp_df.rename(columns={"result": column_name})
            elif "VWAP" in temp_df.columns:
                temp_df = temp_df.rename(columns={"VWAP": column_name})
            else:
                print(f"⚠️ Warning: 'result' or 'VWAP' column not found in {file_path.name}. Using the second column.")
                value_column_name = temp_df.columns[1]
                temp_df = temp_df.rename(columns={value_column_name: column_name})
                
            temp_df["data"] = pd.to_datetime(temp_df["data"])
            
            # Filter by period and select columns
            mask = (temp_df['data'] >= start_date) & (temp_df['data'] <= end_date)
            temp_df = temp_df.loc[mask, ["data", column_name]].set_index('data')
            df_list.append(temp_df)
        except Exception as e:
            print(f"❌ Error processing file {file_path.name}: {e}")

    return pd.concat(df_list, axis=1)

def load_local_variable(variable_name, directory_path, start_date, end_date):
    """
    Loads a specific variable's CSV file from the local data directory.
    
    Args:
        variable_name (str): Name of the variable (and the .csv file).
        directory_path (Path): Path object for the 'dados_api' directory.
        start_date (datetime): Start date for filtering.
        end_date (datetime): End date for filtering.

    Returns:
        pd.DataFrame: DataFrame containing the variable's time series.
    """
    file_path = directory_path / f"{variable_name}.csv"
    try:
        df = pd.read_csv(file_path, index_col='data', parse_dates=True)
        df = df.rename(columns={df.columns[0]: variable_name}) # Ensure column name
        # Resample to daily frequency and forward-fill missing values
        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(daily_index).ffill()
        # Filter by the analysis period
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask]
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: File not found: {file_path}. The script cannot continue.")
        exit() # Terminate the script if an essential file is not found
    except Exception as e:
        print(f"❌ Error loading variable {variable_name}: {e}")
        exit()


def process_ena_features(ena_hist_df):
    """
    Applies moving averages and lags to the ENA series.
    
    Args:
        ena_hist_df (pd.DataFrame): DataFrame with the historical ENA series.

    Returns:
        pd.DataFrame: DataFrame with the new ENA features.
    """
    print("🧠 Processing ENA features (moving averages and lags)...")
    # Moving Averages
    ma_windows = [5, 7, 15, 30, 45, 60, 90]
    ma_dfs = [ena_hist_df.rolling(window=w).mean().rename(columns={'ena_hist': f'MA_{w}_ena_hist'}) for w in ma_windows]
    
    # Lags (Shifts)
    shift_days = [30, 60, 90, 120]
    shift_dfs = [ena_hist_df.shift(d).rename(columns={'ena_hist': f'Shift_{d}_ena_hist'}) for d in shift_days]
    
    return pd.concat([ena_hist_df] + ma_dfs + shift_dfs, axis=1)

# ==============================================================================
# 4. STATISTICAL ANALYSIS FUNCTIONS
# ==============================================================================

def calculate_spearman_correlation(window_df):
    """Calculates the Spearman correlation matrix for a DataFrame."""
    corr_matrix, _ = spearmanr(window_df.dropna())
    return pd.DataFrame(corr_matrix, index=window_df.columns, columns=window_df.columns)

def calculate_mutual_information(window_df):
    """Calculates the Mutual Information matrix for a DataFrame."""
    window_df = window_df.dropna()
    mi_scores = pd.DataFrame(index=window_df.columns, columns=window_df.columns)
    for col in window_df.columns:
        X = window_df.drop(col, axis=1)
        y = window_df[col]
        mi = mutual_info_regression(X, y)
        mi_series = pd.Series(mi, index=X.columns)
        for idx, val in mi_series.items():
            mi_scores.loc[col, idx] = val
            mi_scores.loc[idx, col] = val # Ensure symmetry
    return mi_scores.astype(float)

# ==============================================================================
# 5. VISUALIZATION AND SAVING FUNCTIONS
# ==============================================================================

def create_output_directories(*directories):
    """Creates the output directories if they do not exist."""
    print("📂 Checking and creating output directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "var").mkdir(exist_ok=True)
        (directory / "ena").mkdir(exist_ok=True)
        (directory / "geral").mkdir(exist_ok=True)


def save_heatmap(plot_df, base_dir, sub_dir, base_name, title, cmap, figsize):
    """
    Generates and saves a heatmap.
    
    Args:
        plot_df (pd.DataFrame): DataFrame with the data for the heatmap.
        base_dir (Path): Base output directory (e.g., 'CS-linear-interp').
        sub_dir (str): Subdirectory ('var', 'ena', 'geral').
        base_name (str): Base name for the image file.
        title (str): Title of the plot.
        cmap (str): Colormap for the heatmap.
        figsize (tuple): Size of the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=plot_df.T, annot=False, cmap=cmap, yticklabels=1, ax=ax)
    
    # Configure X-axis ticks for better readability
    num_ticks = len(plot_df.index)
    tick_pos = np.linspace(0, num_ticks - 1, 6, dtype=int)
    tick_labels = [plot_df.index[i].strftime('%Y-%m') for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0, horizontalalignment='center')
    
    ax.set_title(title)
    
    # Sanitize filename to be valid on any OS
    safe_filename = base_name.replace('/', '_').replace('\\', '_')
    save_path = base_dir / sub_dir / f"{safe_filename}.png"
    
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig) # Close the figure to free up memory

# ==============================================================================
# 6. MAIN FUNCTION
# ==============================================================================

def main():
    """Main function that orchestrates the entire process."""
    
    # --- Loading all data ---
    print("--- STARTING DATA LOADING STAGE ---")
    market_df = load_market_data(MARKET_DATA_DIR, START_DATE, END_DATE)
    
    api_variable_names = [
        "GPR", "ibc", "pib_mensal", "oni", "tna", "ena_hist",
        "earm_sin", "earm_n", "earm_ne", "earm_s", "earm_se",
        "cmo_sin", "cmo_n", "cmo_ne", "cmo_s", "cmo_se",
        "proj_carga_M0", "proj_carga_M1", "proj_carga_M2", "proj_carga_M3", "proj_carga_M4",
        "restricao_norte", "restricao_sul", "restricao_nordeste"
    ]
    
    api_df_list = [load_local_variable(v, API_DATA_DIR, START_DATE, END_DATE) for v in api_variable_names]
    
    # --- Feature Engineering (ENA) ---
    # Find the ena_hist df in the list and process it
    ena_hist_df = next((df for df in api_df_list if 'ena_hist' in df.columns), None)
    if ena_hist_df is not None:
        processed_ena_df = process_ena_features(ena_hist_df)
        # Remove the original ena_hist to avoid duplication
        api_df_list = [df for df in api_df_list if 'ena_hist' not in df.columns]
        api_df_list.append(processed_ena_df)

    # --- Consolidation into a Single DataFrame ---
    print("Consolidating all data into a single DataFrame...")
    full_df = pd.concat([market_df] + api_df_list, axis=1)
    full_df = full_df.dropna(axis=0, how='all').sort_index()
    print(f"Final DataFrame created with {full_df.shape[0]} rows and {full_df.shape[1]} columns.")
    print(f"Final data period: from {full_df.index.min().date()} to {full_df.index.max().date()}")
    
    # --- Create output directories ---
    create_output_directories(SPEARMAN_OUTPUT_DIR, MI_OUTPUT_DIR)

    # =================================================
    # --- SPEARMAN CORRELATION (CS) ANALYSIS ---
    # =================================================
    print("\n--- STARTING SPEARMAN CORRELATION ANALYSIS (365-DAY WINDOW) ---")
    rolling_correlation = full_df.p_rolling("365D").apply(calculate_spearman_correlation)
    rolling_correlation = rolling_correlation.dropna()

    # Extract and organize the results
    cs_results = {col: pd.DataFrame([m.loc[col] for m in rolling_correlation], index=rolling_correlation.index) for col in full_df.columns}
    
    print("Generating and saving Spearman Correlation heatmaps...")
    # Group columns for plotting
    market_cols = market_df.columns
    ena_cols = processed_ena_df.columns
    general_cols = [c for c in full_df.columns if c not in market_cols and c not in ena_cols]

    for column_name in full_df.columns:
        result_df = cs_results[column_name]
        # Save for each group of variables
        save_heatmap(result_df[market_cols], SPEARMAN_OUTPUT_DIR, "var", column_name, f'Spearman Correlation of {column_name} vs Market Variables', "PiYG", SPEARMAN_FIGSIZE)
        save_heatmap(result_df[ena_cols], SPEARMAN_OUTPUT_DIR, "ena", column_name, f'Spearman Correlation of {column_name} vs ENA Variables', "PiYG", SPEARMAN_FIGSIZE)
        save_heatmap(result_df[general_cols], SPEARMAN_OUTPUT_DIR, "geral", column_name, f'Spearman Correlation of {column_name} vs General Variables', "PiYG", SPEARMAN_FIGSIZE)
    print("✅ Spearman Correlation heatmaps saved successfully!")

    # =================================================
    # --- MUTUAL INFORMATION (MI) ANALYSIS ---
    # =================================================
    print("\n--- STARTING MUTUAL INFORMATION ANALYSIS (365-DAY WINDOW) ---")
    rolling_mi = full_df.p_rolling("365D").apply(calculate_mutual_information)
    rolling_mi = rolling_mi.dropna()

    # Extract and organize the results
    mi_results = {col: pd.DataFrame([m.loc[col] for m in rolling_mi], index=rolling_mi.index) for col in full_df.columns}

    print("Generating and saving Mutual Information heatmaps...")
    for column_name in full_df.columns:
        result_df = mi_results[column_name]
        # Save for each group of variables
        save_heatmap(result_df[market_cols], MI_OUTPUT_DIR, "var", column_name, f'Mutual Information of {column_name} vs Market Variables', "YlGnBu", MI_FIGSIZE)
        save_heatmap(result_df[ena_cols], MI_OUTPUT_DIR, "ena", column_name, f'Mutual Information of {column_name} vs ENA Variables', "YlGnBu", MI_FIGSIZE)
        save_heatmap(result_df[general_cols], MI_OUTPUT_DIR, "geral", column_name, f'Mutual Information of {column_name} vs General Variables', "YlGnBu", MI_FIGSIZE)
    print("✅ Mutual Information heatmaps saved successfully!")

    print("\n🎉 Analysis completed successfully!")


if __name__ == "__main__":
    main()