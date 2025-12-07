
# -----------------------------------------------------
# Import necessary Library
# -----------------------------------------------------
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import zipfile
import rarfile
from io import StringIO


# -----------------------------------------------------
# Download Data From Web Site
# -----------------------------------------------------
def download_filtered_files(base_url, target_url, download_dir, min_year=2016):
    """
    Downloads files from a target URL when their filename contains a year > min_year.
    The filename structure is expected to contain a year like *_2018 or *-2020.
    
    Parameters:
        base_url (str): Base website URL
        target_url (str): Web page containing file links
        download_dir (str): Local directory to save files
        min_year (int): Minimum year threshold (default: 2016)
    """
    
    # Valid extensions to download
    EXTENSIONS = ('.xlsx', '.csv', '.txt', '.zip', '.rar')

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Create session
    session = requests.Session()
    response = session.get(target_url)

    if response.status_code != 200:
        print("[✖] Failed to access the target page.")
        return

    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    download_links = []

    # Extract download links
    for link in links:
        href = link['href']
        if any(href.lower().endswith(ext) for ext in EXTENSIONS):
            full_url = urljoin(base_url, href)
            download_links.append(full_url)

    print(f"[+] {len(download_links)} files detected (before filtering).")

    # Download only files with year > min_year
    for file_url in download_links:
        file_name = file_url.split("/")[-1].split("?")[0]

        # Extract year from filename (numbers between 1900-2099)
        match = re.search(r"(19|20)\d{2}", file_name)
        if not match:
            # Skip files without identifiable year
            continue

        year = int(match.group())

        if year <= min_year:
            # Skip old files
            continue

        dest_path = os.path.join(download_dir, file_name)

        print(f"[↓] Downloading: {file_name} (year = {year})")
        file_data = session.get(file_url)

        with open(dest_path, 'wb') as f:
            f.write(file_data.content)

    print("[✔] Download completed with year filtering.")


# -----------------------------------------------------
# Extract Zip/Rar File
# -----------------------------------------------------
def extract_files(input_dir, extract_dir):
    """
    Extracts or copies files from input_dir into extract_dir.
    
    - Copies .csv, .txt, .xls, .xlsx
    - Extracts .zip and .rar archives (only CSV/TXT inside)
    - Detects real archive type using magic bytes
    
    Parameters:
        input_dir (str): Directory containing downloaded files
        extract_dir (str): Directory to store extracted/copied files
    """

    os.makedirs(extract_dir, exist_ok=True)

    def detect_real_archive_type(file_path):
        try:
            with open(file_path, 'rb') as f:
                magic_bytes = f.read(4)
            if magic_bytes.startswith(b'PK\x03\x04'):
                return 'zip'
            elif magic_bytes.startswith(b'Rar!'):
                return 'rar'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_name)[1].lower()

        # COPY simple files
        if ext in (".csv", ".txt", ".xls", ".xlsx"):
            dest_path = os.path.join(extract_dir, file_name)
            with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            print(f"[✔] Fichier copié : {file_name}")

        # PROCESS ARCHIVES
        elif ext in (".zip", ".rar"):
            real_type = detect_real_archive_type(file_path)

            try:
                if real_type == "zip":
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        for inner_file in zip_ref.namelist():
                            if inner_file.lower().endswith((".csv", ".txt")):
                                content = zip_ref.read(inner_file)
                                dest_path = os.path.join(extract_dir, os.path.basename(inner_file))
                                with open(dest_path, 'wb') as f_out:
                                    f_out.write(content)
                                print(f"[✔] ZIP : {inner_file} extrait")

                elif real_type == "rar":
                    with rarfile.RarFile(file_path) as rar:
                        for inner_file in rar.namelist():
                            if inner_file.lower().endswith((".csv", ".txt")):
                                content = rar.read(inner_file)
                                dest_path = os.path.join(extract_dir, os.path.basename(inner_file))
                                with open(dest_path, 'wb') as f_out:
                                    f_out.write(content)
                                print(f"[✔] RAR : {inner_file} extrait")

                else:
                    print(f"[✖] Type d’archive non reconnu : {file_name}")

            except Exception as e:
                print(f"[✖] Erreur lors du traitement de {file_name} : {e}")

        # IGNORE OTHER FILE TYPES
        else:
            print(f"[!] Fichier ignoré : {file_name} (extension inconnue)")


# -----------------------------------------------------
# Read CSV/TXT File
# -----------------------------------------------------

def load_dataframes_from_folder(folder_path):
    """
    Reads all CSV and TXT (fixed-width) files inside a folder.
    Returns a dictionary { filename : DataFrame }
    """

    # ----------------------------
    # Detect positions of fixed-width columns from dashed line
    # ----------------------------
    def detect_column_positions_from_dashes(line):
        positions = []
        in_dash = False
        for i, char in enumerate(line):
            if char == '-':
                if not in_dash:
                    start = i
                    in_dash = True
            elif in_dash:
                end = i
                positions.append((start, end))
                in_dash = False
        if in_dash:
            positions.append((start, len(line)))
        return positions

    # ----------------------------
    # TXT Fixed-width File Reader (Improved)
    # ----------------------------
    def read_fixed_width_file(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1️⃣ Remove leading empty or whitespace-only lines
        cleaned_lines = [ln for ln in lines if ln.strip() != ""]

        if len(cleaned_lines) < 3:
            raise ValueError("File too short or malformed.")

        # 2️⃣ Detect dashed separator line
        dash_line_index = -1
        for i, line in enumerate(cleaned_lines[:10]):  # inspect first 10 lines
            stripped = line.strip()
            if stripped != "" and all(c in "- " for c in stripped):
                dash_line_index = i
                break

        if dash_line_index <= 0:
            raise ValueError("Separator '-----' not found.")

        dash_line = cleaned_lines[dash_line_index]

        # 3️⃣ Detect column boundaries
        colspecs = detect_column_positions_from_dashes(dash_line)

        # 4️⃣ Extract headers from line above dashed separator
        headers_line = cleaned_lines[dash_line_index - 1]
        headers = [headers_line[start:end].strip() for (start, end) in colspecs]

        # 5️⃣ Combine data lines into DataFrame
        data_lines = cleaned_lines[dash_line_index + 1:]
        data_str = "".join(data_lines)
        df = pd.read_fwf(StringIO(data_str), colspecs=colspecs, names=headers)

        return df

    # ----------------------------
    # Main loop — Load .csv and .txt
    # ----------------------------
    dataframes = {}

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        # CSV --------------------------------------------------------
        if file.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path, sep=None, engine="python")
                dataframes[file] = df
                print(f"📄 CSV OK : {file} — {df.shape}")
            except Exception as e:
                print(f"❌ Erreur CSV {file}: {e}")

        # TXT --------------------------------------------------------
        elif file.lower().endswith(".txt"):
            try:
                df = read_fixed_width_file(file_path)
                dataframes[file] = df
                print(f"📄 TXT OK : {file} — {df.shape}")
            except Exception as e:
                print(f"❌ Erreur TXT {file}: {e}")

        else:
            print(f"⏭️ Ignoré : {file}")

    return dataframes


# -----------------------------------------------------
# Groups Data Into Unified Dataframe
# -----------------------------------------------------

def group_histo_segments(dataframes_dict):
    """
    Groups histo_cotation_{year} and histo_indice_{year} files into 
    two unified DataFrames keeping ONLY columns common to all years.
    
    Returns:
        df_cotation, df_indice
    """

    cotation_dfs = []
    indice_dfs = []

    pattern = r"(histo_cotation|histo_indice).*?(20\d{2})"

    for filename, df in dataframes_dict.items():

        match = re.search(pattern, filename.lower())
        if match:
            segment_type = match.group(1)
            year = int(match.group(2))

            df.columns = [c.strip() for c in df.columns]

            if segment_type == "histo_cotation":
                cotation_dfs.append(df)

            elif segment_type == "histo_indice":
                indice_dfs.append(df)

    # ---------------------------------------
    # 1️⃣ Intersection of columns (common to ALL)
    # ---------------------------------------

    def concat_with_common_columns(df_list):
        if not df_list:
            return pd.DataFrame()

        # Compute intersection of all columns
        common_cols = set(df_list[0].columns)
        for d in df_list[1:]:
            common_cols &= set(d.columns)

        common_cols = sorted(list(common_cols))  # sorted for consistency

        # Keep only these columns in each DF
        reduced_dfs = [d[common_cols].copy() for d in df_list]

        # Concatenate
        return pd.concat(reduced_dfs, ignore_index=True)

    # Final DFs
    df_cotation = concat_with_common_columns(cotation_dfs)
    df_indice = concat_with_common_columns(indice_dfs)

    return df_cotation, df_indice


# -----------------------------------------------------
# Build Indexes and Storck Price Dataframe
# -----------------------------------------------------

def build_cotation_indice(all_dataframes):
    """
    Build cotation and indice final DataFrames using:
    - explicit column map for cotation
    - intersection of identical columns for indice
    """

    # Explicit mapping for cotation (your mapping)
    cotation_map = {
        "GROUPE": "C_GR_RLC",
        "CODE": "CODE_VAL",
        "VALEUR": "LIB_VAL",
        "NB_TRANSACTION": "NB_TRAN"
    }

    def make_unique_columns(columns):
        """Ensure no duplicate column names exist."""
        counts = {}
        new_cols = []
        for col in columns:
            if col not in counts:
                counts[col] = 0
                new_cols.append(col)
            else:
                counts[col] += 1
                new_cols.append(f"{col}.{counts[col]}")
        return new_cols

    def apply_cotation_mapping(df):
        """Apply explicit renaming only when needed."""
        rename_dict = {col: cotation_map[col] for col in df.columns if col in cotation_map}
        df = df.rename(columns=rename_dict)
        df.columns = make_unique_columns(df.columns)
        return df

    def unify_and_concat(dfs, apply_map=False):
        """Standard concatenation logic with optional mapping."""
        if not dfs:
            return pd.DataFrame()

        df_list = []
        for df in dfs:
            df = df.dropna(how="all").reset_index(drop=True)
            if apply_map:
                df = apply_cotation_mapping(df)
            df.columns = make_unique_columns(df.columns)
            df_list.append(df)

        # keep only columns present in ALL dataframes
        common_cols = list(set.intersection(*(set(df.columns) for df in df_list)))
        common_cols.sort()

        return pd.concat([df[common_cols] for df in df_list], ignore_index=True)

    # ----------------------------------------------
    # Split dataframes into cotation and indice
    # ----------------------------------------------
    dfs_cotation = [v for k, v in all_dataframes.items() if "cotation" in k.lower()]
    dfs_indice   = [v for k, v in all_dataframes.items() if "indice" in k.lower()]

    # ----------------------------------------------
    # Build final outputs
    # ----------------------------------------------
    df_cotation_final = unify_and_concat(dfs_cotation, apply_map=True)
    df_indice_final   = unify_and_concat(dfs_indice, apply_map=False)

    return df_cotation_final, df_indice_final


# -----------------------------------------------------
# COLORS
# -----------------------------------------------------

def extract_subdf(df, column, values):
    """
    Extract a sub-dataframe where `column` is in the list `values`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name on which the filter is applied.
    values : list, set, tuple
        List of modalities (values) to retain.
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    # Ensure values is a list-like object
    if not isinstance(values, (list, set, tuple)):
        values = [values]

    return df[df[column].isin(values)].copy()


# -----------------------------------------------------
# Save Dataframe to Excel File
# -----------------------------------------------------

def save_dataframes_to_excel(folder_save, dataframes, max_rows=1_048_000):
    """
    Save one or multiple DataFrames to Excel files.
    Automatically splits files if row count exceeds Excel limit.
    """

    os.makedirs(folder_save, exist_ok=True)

    # If we receive a list, convert to named dict
    if isinstance(dataframes, list):
        dataframes = {f"df_{i+1}": df for i, df in enumerate(dataframes)}

    for name, df in dataframes.items():

        n_rows = len(df)

        if n_rows <= max_rows:
            # Normal save
            file_path = os.path.join(folder_save, f"{name}.xlsx")
            df.to_excel(file_path, index=False)
            print(f"Saved → {file_path}")
        else:
            # Split into chunks of max_rows
            n_files = (n_rows // max_rows) + 1

            print(f"{name} is too large ({n_rows} rows). Splitting into {n_files} files...")

            for i in range(n_files):
                chunk = df.iloc[i*max_rows:(i+1)*max_rows]
                file_path = os.path.join(folder_save, f"{name}_part{i+1}.xlsx")
                chunk.to_excel(file_path, index=False)
                print(f"Saved chunk → {file_path}")

    print("\nDone. All DataFrames saved successfully.")
