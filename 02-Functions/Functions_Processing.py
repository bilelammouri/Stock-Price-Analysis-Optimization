import os
import pandas as pd  
import numpy as np
import re
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

# -----------------------------------------------------
# COLORS
# -----------------------------------------------------
custom_blue   = "#0052A4"   
custom_orange = "#ED6B2E"    


# -------------------------------------------------------------
# Read Files From folder
# -------------------------------------------------------------

def read_split_excel(folder_path, base_name):
    """
    Reads Excel files that may be:
      - base_name.xlsx
      - base_name_part1.xlsx, base_name_part2.xlsx, ...
    Returns a single DataFrame.
    """
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and not f.startswith("~$")]
    
    # Pattern for split parts
    pattern = re.compile(rf"{base_name}_part(\d+)\.xlsx")
    parts = []

    for f in files:
        match = pattern.match(f)
        if match:
            part_no = int(match.group(1))
            parts.append((part_no, f))

    # CASE 1 — Split files found
    if parts:
        parts = sorted(parts, key=lambda x: x[0])
        print(f"Detected {len(parts)} split parts for '{base_name}'.")
        
        dfs = []
        for _, fname in parts:
            print(f"Loading {fname} ...")
            full_path = os.path.join(folder_path, fname)
            dfs.append(pd.read_excel(full_path))

        dfFinal = pd.concat(dfs, ignore_index=True)
        print(f"→ Final shape: {dfFinal.shape}")
        return dfFinal

    # CASE 2 — Single file exists
    single_file = f"{base_name}.xlsx"
    if single_file in files:
        print(f"Loading single file: {single_file}")
        return pd.read_excel(os.path.join(folder_path, single_file))

    # CASE 3 — No matching file found
    raise FileNotFoundError(f"No Excel files found for '{base_name}' (neither split nor single).")


# -------------------------------------------------------------
# Clean Stock Price and Indeces Dataframe
# -------------------------------------------------------------

def clean_stock_df(df, num_cols, date_col="SEANCE", cat_cols=None,
                   missing_strategy="drop", fill_value=None):
    """
    missing_strategy:
        - "drop"
        - "mean"
        - "median"
        - "min"
        - "max"
        - "std"
        - "value"  (requires fill_value)
    """

    df = df.copy()

    # 1️⃣ Clean numeric columns: convert to numeric
    for col in num_cols:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2️⃣ Handle missing values depending on the chosen strategy
    if missing_strategy == "drop":
        df = df.dropna(subset=num_cols, how="any")

    else:
        for col in num_cols:
            if missing_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())

            elif missing_strategy == "median":
                df[col] = df[col].fillna(df[col].median())

            elif missing_strategy == "min":
                df[col] = df[col].fillna(df[col].min())

            elif missing_strategy == "max":
                df[col] = df[col].fillna(df[col].max())

            elif missing_strategy == "std":
                df[col] = df[col].fillna(df[col].std())

            elif missing_strategy == "value":
                if fill_value is None:
                    raise ValueError("missing_strategy='value' requires fill_value.")
                df[col] = df[col].fillna(fill_value)

            else:
                raise ValueError("Unknown missing_strategy option.")

    # 3️⃣ Convert date column
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    # 4️⃣ Clean categorical columns
    if cat_cols:
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()

    return df


# -------------------------------------------------------------
# Summary Table
# -------------------------------------------------------------

def summary_table(df):

    rows = []

    for col in df.columns:
        data = df[col]
        dtype = data.dtype
        missing = data.isna().sum()
        unique = data.nunique()

        # -------------------------------------
        # 1️⃣ Numeric columns
        # -------------------------------------
        if pd.api.types.is_numeric_dtype(dtype):
            desc = {
                "count": data.count(),
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "25%": data.quantile(0.25),
                "50%": data.quantile(0.50),
                "75%": data.quantile(0.75),
                "max": data.max(),
                "skewness": data.skew(),
                "kurtosis": data.kurtosis(),
                "missing": missing,
                "unique": unique,
                "dtype": str(dtype)
            }

        # -------------------------------------
        # 2️⃣ Datetime columns
        # -------------------------------------
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            desc = {
                "count": data.count(),
                "mean": None,
                "std": None,
                "min": data.min(),
                "25%": None,
                "50%": None,
                "75%": None,
                "max": data.max(),
                "skewness": None,
                "kurtosis": None,
                "missing": missing,
                "unique": unique,
                "dtype": str(dtype)
            }

        # -------------------------------------
        # 3️⃣ Categorical / Object columns
        # -------------------------------------
        else:
            top = data.mode().iloc[0] if not data.mode().empty else None
            freq = data.value_counts().max() if not data.value_counts().empty else None

            desc = {
                "count": data.count(),
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None,
                "skewness": None,
                "kurtosis": None,
                "missing": missing,
                "unique": unique,
                "dtype": str(dtype)
            }

        desc["variable"] = col
        rows.append(desc)

    # Build DataFrame
    result = pd.DataFrame(rows).set_index("variable")

    # Replace NaN with "-"
    result = result.replace({np.nan: "-"})

    return result


# -------------------------------------------------------------
# Compute Adjustment Coefficient
# -------------------------------------------------------------

def compute_adjustment_coefficient(df_price, df_div):
    """
    Compute adjustment coefficients for each stock using dividend data.
    Assumes:
    - df_price.index = SEANCE (dates)
    - df_div.index = aligned to df_price (same dates)
    """
    df_price = df_price.copy().astype(float)
    df_div = df_div.copy().astype(float)

    # Ensure indexes match
    df_div = df_div.reindex(df_price.index).fillna(0)

    # Initialize adjustment coefficient dataframe
    adje_coef = pd.DataFrame(1.0, index=df_price.index, columns=df_price.columns)

    # Loop over each stock
    for col in df_price.columns:
        for t in range(1, len(df_price)):
            P_prev = df_price[col].iloc[t-1]
            D_curr = df_div[col].iloc[t]

            # Skip if previous price is NaN
            if pd.isna(P_prev):
                adje_coef[col].iloc[t] = adje_coef[col].iloc[t-1]
                continue

            # Compute adjustment factor
            adje_coef[col].iloc[t] = (P_prev / (P_prev - D_curr)) * adje_coef[col].iloc[t-1]

    return adje_coef


# -------------------------------------------------------------
# Generate Adjusted Price
# -------------------------------------------------------------

def compute_adjusted_prices(df_price, df_div):
    """
    Compute adjusted stock prices using dividends.
    Assumes:
    - df_price.index = SEANCE dates (datetime)
    - df_div.index = dividend dates (datetime)
    """
    df_adj = df_price.copy().astype(float)

    # Ensure index is datetime
    df_price.index = pd.to_datetime(df_price.index)
    df_div.index = pd.to_datetime(df_div.index)

    # Reindex dividend data to match price dates; missing dates = 0
    df_div_aligned = df_div.reindex(df_price.index).fillna(0)

    # Add missing dividend columns
    for col in df_price.columns:
        if col not in df_div_aligned.columns:
            df_div_aligned[col] = 0

    # Compute adjusted prices
    for col in df_price.columns:
        price = df_price[col]
        div = df_div_aligned[col]

        for i in range(1, len(price)):
            D = div.iloc[i]
            P_prev = price.iloc[i - 1]

            if D == 0 or pd.isna(P_prev):
                continue

            factor = (P_prev - D) / P_prev
            df_adj.loc[:price.index[i], col] *= factor

    return df_adj


# -------------------------------------------------------------
# Compute Returns values
# -------------------------------------------------------------

def compute_returns(df, return_type="log", freq="D", dropna=True, suffix="_ret"):
    """
    Compute returns (log or simple) at any frequency: daily, weekly, monthly, yearly.

    Parameters
    ----------
    df : DataFrame
        Price dataframe indexed by datetime.
        
    return_type : str
        'log'    -> natural log return
        'simple' -> pct change
        
    freq : str
        'D' = daily (no resampling)
        'W' = weekly
        'M' = monthly
        'Y' = yearly
        
    dropna : bool
        Drop missing values after computing returns.
        
    suffix : str
        Suffix appended to return columns.
        
    Returns
    -------
    DataFrame of returns
    """
        
    # Ensure datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Keep only numeric columns
    price_df = df.select_dtypes(include=["number"])
    
    # ---- Frequency management ----
    if freq.upper() == "D":
        data_f = price_df
    else:
        data_f = price_df.resample(freq.upper()).last()
    
    # ---- Compute returns ----
    if return_type.lower() == "log":
        ret = np.log(data_f / data_f.shift(1))     # natural log
    elif return_type.lower() == "simple":
        ret = data_f.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'simple'")
    
    # Rename columns
    ret.columns = [col + suffix for col in ret.columns]
    
    # Drop NA if required
    if dropna:
        ret = ret.fillna(0)
    
    return ret


# -------------------------------------------------------------
# Compute Cumulative Returns
# -------------------------------------------------------------

def compute_cumulative_returns(df_returns):
    """
    Compute cumulative returns for all columns in a DataFrame of returns.
    """
    cum_ret = (1 + df_returns).cumprod() - 1
    return cum_ret


# -------------------------------------------------------------
# Compute drawdowns
# -------------------------------------------------------------

def compute_drawdowns(cum_returns):
    """
    cum_returns: Series of simple returns.
    Returns DataFrame: wealth, peak, drawdown.
    """
    wealth = (1 + cum_returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak

    return pd.DataFrame({"wealth": wealth, "peak": peak, "drawdown": drawdown})


# -------------------------------------------------------------
# Expand Dictionnary
# -------------------------------------------------------------

def expand_sector_dict(group_dict):
    """
    Convert {sector: [list of stocks]} → {stock: sector}
    """
    out = {}
    for sector, stocks in group_dict.items():
        for s in stocks:
            out[s] = sector
    return out


# -------------------------------------------------------------
# Sector → tickers map
# -------------------------------------------------------------

sub_sector_map = {
    "STES FINANCIERES" : ["BTE (ADP)", "ATTIJARI BANK", "BIAT", "BH BANK", 
                          "BT", "UBCI", "STB", "BNA", "AMEN BANK", "ATB", "UIB", 
                          "WIFACK INT BANK", "ASTREE", "STAR", 
                          "BH ASSURANCE", "TUNIS RE", "ASSUR MAGHREBIA", 
                          "ASSU MAGHREBIA VIE", "SPDIT - SICAF", "TUNISIE LEASING F",
                          "PLAC. TSIE-SICAF", "TUNINVEST-SICAR", "CIL", "ATL", 
                          "ATTIJARI LEASING", "BH LEASING", "HANNIBAL LEASE", 
                          "BEST LEASE"],
    "BANQUES" : ["BTE (ADP)", "ATTIJARI BANK", "BIAT", "BH BANK", "BT", "UBCI", 
                 "STB", "BNA", "AMEN BANK", "ATB", "UIB", "WIFACK INT BANK"],
    "ASSURANCES" : ["ASTREE", "STAR", "BH ASSURANCE", "TUNIS RE", 
                    "ASSUR MAGHREBIA", "ASSU MAGHREBIA VIE"],
    "SERVICES FINANCIER" : ["SPDIT - SICAF", "TUNISIE LEASING F", 
                            "PLAC. TSIE-SICAF", "TUNINVEST-SICAR", "CIL", 
                            "ATL", "ATTIJARI LEASING", "BH LEASING", 
                            "HANNIBAL LEASE", "BEST LEASE"],
    "SERVICES AUX CONSO" : ["MONOPRIX", "MAGASIN GENERAL", "SOTUMAG", "ARTES", 
                            "ENNAKL AUTOMOBILES", "CITY CARS", "SMART TUNISIE", 
                            "STA"],
    "DISTRIBUTION" : ["MONOPRIX", "MAGASIN GENERAL", "SOTUMAG", "ARTES", 
                      "ENNAKL AUTOMOBILES", "CITY CARS", "SMART TUNISIE", "STA"],
    "AGROALIMENT BOISSO" : ["SFBT", "POULINA GP HOLDING", "LAND OR", 
                            "DELICE HOLDING"],
    "PROD MENAGER SOIN" : ["NEW BODY LINE", "EURO-CYCLES", "SAH", "OFFICEPLAST", 
                           "ATELIER MEUBLE INT"],
    "INDUSTRIES" : ["SIMPAR", "SOMOCER", "SITS", "ESSOUKNA", 
                    "CIMENTS DE BIZERTE", "CARTHAGE CEMENT", "SOTEMAIL", 
                    "MPBS", "SOTUVER", "SIAME", "ONE TECH HOLDING"],
    "BATIM MATE CONSTRU" : ["SIMPAR", "SOMOCER", "SITS", "ESSOUKNA", 
                            "CIMENTS DE BIZERTE", "CARTHAGE CEMENT", "SOTEMAIL", 
                            "MPBS"],
    "MATERIAUX DE BASE" : ["AIR LIQUIDE TSIE", "ICF", 
                           "SOCIETE TUNISIE PROFILES ALUMINIUM", 
                           "SOTIPAPIER"]}

Actual_portfolio = ["POULINA GP HOLDING", "SFBT", "DELICE HOLDING", "LAND OR",
                    "SAH", "NEW BODY LINE", "CARTHAGE CEMENT", "MPBS", "ATL",
                    "SOTEMAIL", "CIMENTS DE BIZERTE", "ONE TECH HOLDING", 
                    "SOTUVER", "TPR", "SOTIPAPIER", "AETECH", "BT", "STB",
                    "TAWASOL GP HOLDING", "AMEN BANK", "BNA", "CITY CARS", 
                    "ENNAKL AUTOMOBILES", "SMART TUNISIE", "ASSUR MAGHREBIA",
                    "STA", "TUNISAIR", "BIAT", "ATTIJARI BANK", "HANNIBAL LEASE",  
                    "ATB", "BTE (ADP)",  "ASSU MAGHREBIA VIE", "BH ASSURANCE"]


TUNINDEX = ["AIR LIQUIDE TSIE", "AMEN BANK", "ARTES", "ASSU MAGHREBIA VIE", 
            "ASSUR MAGHREBIA", "ASTREE", "ATB", "ATELIER MEUBLE INT", "ATL", 
            "ATTIJARI BANK", "ATTIJARI LEASING", "BEST LEASE", "BH ASSURANCE", 
            "BH BANK", "BH LEASING", "BIAT", "BNA", "BT", "BTE (ADP)", 
            "CARTHAGE CEMENT", "CIL", "CIMENTS DE BIZERTE", "CITY CARS", 
            "DELICE HOLDING", "ENNAKL AUTOMOBILES", "ESSOUKNA", "EURO-CYCLES", 
            "HANNIBAL LEASE", "ICF", "LAND OR", "MAGASIN GENERAL", "MONOPRIX", 
            "MPBS", "NEW BODY LINE", "OFFICEPLAST", "ONE TECH HOLDING", 
            "PLAC. TSIE-SICAF", "POULINA GP HOLDING", "SAH", "SFBT", "SIAME", 
            "SIMPAR", "SITS", "SMART TUNISIE", "SOMOCER", "SOTEMAIL", 
            "SOTETEL", "SOTIPAPIER", "SOTRAPIL", "SOTUMAG", "SOTUVER", 
            "SPDIT - SICAF", "STA", "STAR", "STB", "STIP", "TELNET HOLDING", 
            "TPR", "TUNINVEST-SICAR", "TUNIS RE", "TUNISIE LEASING F", "UBCI", 
            "UIB", "UNIMED"]

TUNINDEX20 = ["AMEN BANK", "ASSU MAGHREBIA VIE", "ATB", "ATTIJARI BANK", 
              "BH BANK", "BIAT", "BNA", "BT", "CARTHAGE CEMENT", "CITY CARS", 
              "DELICE HOLDING", "EURO-CYCLES", "ONE TECH HOLDING", 
              "POULINA GP HOLDING", "SAH", "SFBT", "SMART TUNISIE", "SOTUVER", 
              "TPR", "UIB"]

new_label_sect = {
    "STES FINANCIERES" : "Financials",
    "BANQUES" :  "Banks",
    "ASSURANCES" : "Insurance",
    "SERVICES FINANCIER" : "Financial Services",
    "SERVICES AUX CONSO" :   "Consumer Services",
    "DISTRIBUTION" : "Retail & Distribution",
    "BIENS CONSOMMATION" : "Consumer Goods",
    "AGROALIMENT BOISSO" : "Food & Beverages",
    "PROD MENAGER SOIN" : "Household & Personal Products",
    "INDUSTRIES" : "Industrials",
    "BATIM MATE CONSTRU" : "Construction & Building Materials",
    "MATERIAUX DE BASE" : "Raw Materials"
    }


