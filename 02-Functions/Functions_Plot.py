import os
import pandas as pd  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

# -----------------------------------------------------
# COLORS
# -----------------------------------------------------
custom_blue   = "#0052A4"   
custom_orange = "#ED6B2E"    

# -------------------------------------------------------------
#  Bivariate Plot --Curve--
# -------------------------------------------------------------

def plot_raw_vs_adjusted(df_raw, df_adj, stock):
    """
    Plot Raw vs Adjusted Price in the same figure.
    """
   
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(custom_orange)  # background

    # Plot Raw and Adjusted
    ax.plot(df_raw.index, df_raw[stock], label='Raw Price', 
            color=custom_blue, linewidth=1, linestyle='-')
    ax.plot(df_adj.index, df_adj[stock], label='Adjusted Price', 
            color=custom_blue, linewidth=2, linestyle='-.')

    # Title and labels
    ax.set_title(f'Raw vs Adjusted Price – {stock}', fontsize=16, 
                 weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Grid and x-axis formatting
    ax.grid(True, color=custom_orange, alpha=0.3)
    ax.tick_params(axis='x', rotation=90)

    # Legend
    ax.legend()

    # Show figure
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Univariate Plot --Curve--
# -------------------------------------------------------------

def plot_stock_graphs(df, save_path, cols=None, show_stock=None):
    """
    Create and save graphs for each selected column.
    Show only the figure for show_stock.
    """

    os.makedirs(save_path, exist_ok=True)

    if not np.issubdtype(df.index.dtype, np.datetime64):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if cols is None:
        selected_cols = list(df.columns)
    elif isinstance(cols, int):
        selected_cols = list(df.columns[:cols])
    elif isinstance(cols, (list, tuple)):
        selected_cols = []
        for c in cols:
            selected_cols.append(df.columns[c] if isinstance(c, int) else c)
    else:
        raise ValueError("cols must be None, int, or list")

    df_returns = df.pct_change()

    for col in selected_cols:

        price = pd.to_numeric(df[col], errors="coerce")

        rs = (
            pd.to_numeric(df_returns[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(custom_orange)

        for ax in axes:
            ax.grid(True, color=custom_orange, alpha=0.35, linewidth=1)

        # PRICE PLOT
        ax0 = axes[0]
        ax0.plot(df.index, price, color=custom_blue, linewidth=2)
        ax0.set_title(f"Price Series — {col}", fontsize=13, weight="bold")
        ax0.tick_params(axis="x", rotation=60)
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Price")

        for item in ([ax0.title, ax0.xaxis.label, ax0.yaxis.label] +
                     ax0.get_xticklabels() + ax0.get_yticklabels()):
            item.set_color("white")

        # DENSITY PLOT
        ax1 = axes[1]

        if rs.size < 3 or rs.nunique() == 1:
            ax1.text(
                0.5, 0.5,
                "Not enough valid return data\nto plot density",
                ha="center", va="center", fontsize=12, color="white",
                transform=ax1.transAxes
            )
            ax1.set_axis_off()
        else:
            try:
                sns.kdeplot(
                    rs, ax=ax1, fill=True,
                    color=custom_blue, alpha=0.7, linewidth=1.5
                )

                ax1.hist(rs, bins=40, density=True,
                         alpha=0.25, color="white", edgecolor="none")

                mu, sigma = stats.norm.fit(rs)
                x = np.linspace(rs.min(), rs.max(), 300)
                ax1.plot(x, stats.norm.pdf(x, mu, sigma),
                         color="black", linestyle="--", linewidth=1.4)

                ax1.set_title(f"Density of Daily Returns — {col}",
                              fontsize=13, weight="bold")

                ax1.set_xlabel("Daily Return")
                ax1.set_ylabel("Density")

                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_color("white")

            except Exception as e:
                ax1.text(
                    0.5, 0.5,
                    f"Error plotting density:\n{str(e)}",
                    ha="center", va="center", fontsize=12,
                    color="white", transform=ax1.transAxes
                )
                ax1.set_axis_off()

        # SAVE + SHOW logic
        out_file = os.path.join(save_path, f"{col}_graph.png")
        plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")

        if show_stock is not None and col == show_stock:
            plt.show()
        else:
            plt.close(fig)

    print(f"Generated {len(selected_cols)} plots. Displayed only: {show_stock}")


# -------------------------------------------------------------
# Multivariate Plot --BoxPlot--
# -------------------------------------------------------------

def plot_boxplot(df_pivot, folder_path, df_name="dataframe",
                 highlight_list=None, highlight_color="red"):
    """
    Plot and save a boxplot for all columns, highlighting selected columns
    in a different color.

    Parameters
    ----------
    df_pivot : DataFrame
        Dataframe of stock prices.

    folder_path : str
        Directory where the plot will be saved.

    df_name : str
        Name used in the saved file.

    highlight_list : list or None
        List of column names to highlight. Optional.

    highlight_color : str
        Color used to highlight selected columns.
    """


    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    cols = df_pivot.columns.tolist()

    # Prepare colors for each box
    colors = []
    for col in cols:
        if highlight_list is not None and col in highlight_list:
            colors.append(highlight_color)      # highlighted column
        else:
            colors.append(custom_blue)          # normal column

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(custom_orange)

    # Boxplot
    box = ax.boxplot(df_pivot[cols].values,
                     labels=cols,
                     patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.4),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black"),
                     flierprops=dict(marker='o', markersize=3, markeredgecolor="gray")
                    )

    # Apply color for each box
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Title and labels
    ax.set_title(f"Boxplot – {df_name}", fontsize=18, weight='bold')
    ax.set_ylabel("Price")

    # Grid
    ax.grid(True, color=custom_orange, alpha=0.3)
    ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()

    # Save figure
    out_path = os.path.join(folder_path, f"{df_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")


# -------------------------------------------------------------
# Multivariate Plot --BoxPlot-- without outliers
# -------------------------------------------------------------

def plot_boxplot_no_outliers(df_pivot, folder_path, df_name="dataframe",
                             highlight_list=None, highlight_color="red"):
    """
    Plot and save a boxplot for all columns, highlighting selected columns,
    without showing outliers.

    Parameters
    ----------
    df_pivot : DataFrame
        Dataframe of stock prices.

    folder_path : str
        Directory where the plot will be saved.

    df_name : str
        Name used in the saved file.

    highlight_list : list or None
        List of column names to highlight. Optional.

    highlight_color : str
        Color used to highlight selected columns.
    """

    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    cols = df_pivot.columns.tolist()

    # Prepare colors for each box
    colors = []
    for col in cols:
        if highlight_list is not None and col in highlight_list:
            colors.append(highlight_color)      # highlighted column
        else:
            colors.append(custom_blue)          # normal column

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(custom_orange)

    # Boxplot without outliers (showfliers=False)
    box = ax.boxplot(df_pivot[cols].values,
                     labels=cols,
                     patch_artist=True,
                     showfliers=False,  # <--- hide outliers
                     medianprops=dict(color="black", linewidth=1.4),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black")
                    )

    # Apply color for each box
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Title and labels
    ax.set_title(f"Boxplot – {df_name}", fontsize=18, weight='bold')
    ax.set_ylabel("Price")

    # Grid
    ax.grid(True, color=custom_orange, alpha=0.3)
    ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()

    # Save figure
    out_path = os.path.join(folder_path, f"{df_name}_no_outliers.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")


# -------------------------------------------------------------
# Mutlivairate Plot --Curve--
# -------------------------------------------------------------

def plot_raw_(df_raw, stock):
    """
    Plot Raw vs Adjusted Price in the same figure.
    """
   
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(custom_orange)  # background

    # Plot Raw and Adjusted
    ax.plot(df_raw.index, df_raw[stock], label='Raw Price', 
            color=custom_blue, linewidth=2, linestyle='-')

    # Title and labels
    ax.set_title(f'Raw vs Adjusted Price – {stock}', fontsize=16, 
                 weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Grid and x-axis formatting
    ax.grid(True, color=custom_orange, alpha=0.3)
    ax.tick_params(axis='x', rotation=90)

    # Legend
    ax.legend()

    # Show figure
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Mutlivairate Plot --Curve--
# -------------------------------------------------------------

def plot_raw_all(df_raw, folder_path, df_name="dataframe"):
    """
    Plot all raw price columns in the same figure.
    """

    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(custom_orange)

    # Plot each stock (each column)
    for col in df_raw.columns:
        ax.plot(df_raw.index, df_raw[col],
                label=col,
                linewidth=1.8)

    # Title and labels
    ax.set_title(f"Raw Prices for All Stocks – {df_name}", fontsize=16, weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Grid and formatting
    ax.grid(True, color=custom_orange, alpha=0.3)
    ax.tick_params(axis='x', rotation=90)

    # Legend outside the plot (clean look)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    # Save figure using dataframe name
    file_path = os.path.join(folder_path, f"{df_name}.png")
    plt.savefig(file_path, dpi=300)
    plt.show() 
    print(f"Saved: {file_path}")


# -------------------------------------------------------------
# Plot drawdown for only ONE selected stock
# -------------------------------------------------------------

def plot_drawdown_single(df_returns, stock):
    """
    Show a drawdown plot for a single stock using fig, ax.
    """
    r = df_returns[stock].fillna(0)
    dd = compute_drawdowns(r)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(custom_orange)

    ax.plot(dd.index, dd["wealth"], label="Wealth", color="blue")
    ax.fill_between(dd.index, dd["peak"], dd["wealth"],
                    color="red", alpha=0.25, label="Drawdown")

    ax.set_title(f"Drawdown Chart — {stock}", fontsize=14, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth Index")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Save drawdown plots for ALL stocks + show one selected
# -------------------------------------------------------------

def save_all_drawdown_plots(df_returns, save_path, show_stock=None):
    """
    For each stock:
        → compute drawdown
        → generate a fig-based plot
        → save PNG in save_path

    show_stock:
        If provided → show only this one, but save all others
    """
    os.makedirs(save_path, exist_ok=True)

    for col in df_returns.columns:
        r = df_returns[col].fillna(0)
        dd = compute_drawdowns(r)

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(custom_orange)

        ax.plot(dd.index, dd["wealth"], label="Wealth", color="blue")
        ax.fill_between(dd.index, dd["peak"], dd["wealth"],
                        color="red", alpha=0.25, label="Drawdown")

        ax.set_title(f"Drawdown Chart — {col}", fontsize=14, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wealth Index")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save plot
        fname = os.path.join(save_path, f"{col}_drawdown.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")

        # Show ONLY selected stock
        if show_stock is not None and col == show_stock:
            plt.show()
        else:
            plt.close(fig)

    print(f"Finished saving drawdown plots → {save_path}")
    if show_stock:
        print(f"Displayed plot for: {show_stock}")


# -------------------------------------------------------------
# Cluster Correlation graph
# -------------------------------------------------------------

def plot_cluster_corr(corr, sector_dict=None, variables=None, figsize=(14, 12),
                      save_dir=None, file_name=None):

    """
    Plot a hierarchical-clustered correlation matrix with optional sector coloring
    and optional variable selection.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    sector_dict : dict, optional
        Mapping: asset → sector.
    variables : list, optional
        Subset of columns (tickers) to plot. Default: all.
    figsize : tuple
        Figure size.
    save_dir : str, optional
        Folder path to save the figure.
    file_name : str, optional
        File name to save the figure.
    """

    # -----------------------------
    # 0. Filter variables
    # -----------------------------
    if variables is not None:
        variables = [v for v in variables if v in corr.columns]
        if len(variables) == 0:
            raise ValueError("None of the specified variables exist in the correlation matrix.")
        corr = corr.loc[variables, variables]

        if sector_dict is not None:
            sector_dict = {k: v for k, v in sector_dict.items() if k in variables}

    corr = corr.copy().fillna(0)

    # -----------------------------
    # 1. Clustering
    # -----------------------------
    link = linkage(corr, method='ward')
    cluster_idx = leaves_list(link)
    corr_clustered = corr.iloc[cluster_idx, cluster_idx]
    labels = corr_clustered.columns

    # -----------------------------
    # 2. Sector colors (optional)
    # -----------------------------
    label_colors = None
    if sector_dict is not None:
        sectors = list({sector_dict.get(col, "Unknown") for col in labels})
        palette = sns.color_palette("tab10", len(sectors))
        sector_to_color = dict(zip(sectors, palette))
        label_colors = [sector_to_color[sector_dict.get(col, "Unknown")]
                        for col in labels]

    # -----------------------------
    # 3. Layout with dendrogram
    # -----------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.25, 1],
        height_ratios=[0.25, 1],
        wspace=0.05, hspace=0.05
    )

    # Top dendrogram
    ax_top = fig.add_subplot(gs[0, 1])
    dendrogram(link, labels=labels, leaf_rotation=90, ax=ax_top)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # Left dendrogram
    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(link, labels=labels, orientation="left", ax=ax_left)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])

    # Colorbar at top-left
    cbar_ax = fig.add_axes([0.07, 0.78, 0.02, 0.15])  # [left, bottom, width, height]

    sns.heatmap(
        corr_clustered,
        cmap="coolwarm",
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_heat,
        cbar=True,
        cbar_ax=cbar_ax
    )

    # Rotate labels
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)  # Z-axis

    # Move Y labels to right
    ax_heat.yaxis.tick_right()
    ax_heat.yaxis.set_label_position("right")

    # Apply sector colors
    if label_colors is not None:
        for tick, color in zip(ax_heat.get_xticklabels(), label_colors):
            tick.set_color(color)
        for tick, color in zip(ax_heat.get_yticklabels(), label_colors):
            tick.set_color(color)

    plt.tight_layout()

    # -----------------------------
    # 4. Save figure
    # -----------------------------
    if save_dir is not None and file_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, file_name)
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {full_path}")

    plt.show()


# -------------------------------------------------------------
# Density Graph
# -------------------------------------------------------------

def plot_stock_density(stock_df, return_df, save_path, show_stock=None):
    """
    Plot stock prices and return distributions for all columns.
    Inputs:
        stock_df : pd.DataFrame - stock prices (index = Date)
        return_df : pd.DataFrame - stock returns (same columns as stock_df)
        save_path : str - folder to save figures
        show_stock : str or None - which stock to display
    """

    os.makedirs(save_path, exist_ok=True)

    # Ensure datetime index
    if not np.issubdtype(stock_df.index.dtype, np.datetime64):
        stock_df = stock_df.copy()
        stock_df.index = pd.to_datetime(stock_df.index)

    for col in stock_df.columns:

        if col not in return_df.columns:
            continue  # skip if return missing

        price = pd.to_numeric(stock_df[col], errors="coerce")
        rs = (
            pd.to_numeric(return_df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(custom_orange)

        for ax in axes:
            ax.grid(True, color=custom_orange, alpha=0.35, linewidth=1)

        # ---------- PRICE PLOT ----------
        ax0 = axes[0]
        ax0.plot(stock_df.index, price, color=custom_blue, linewidth=2)
        ax0.set_title(f"Price Series — {col}", fontsize=13, weight="bold")
        ax0.tick_params(axis="x", rotation=60)
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Price")
        for item in ([ax0.title, ax0.xaxis.label, ax0.yaxis.label] +
                     ax0.get_xticklabels() + ax0.get_yticklabels()):
            item.set_color("white")

        # ---------- RETURN DISTRIBUTION ----------
        ax1 = axes[1]
        if len(rs) < 3 or rs.nunique() == 1:
            ax1.text(
                0.5, 0.5,
                "Not enough valid return data\nto plot density",
                ha="center", va="center", fontsize=12, color="white",
                transform=ax1.transAxes
            )
            ax1.set_axis_off()
        else:
            try:
                sns.kdeplot(rs, ax=ax1, fill=True, color=custom_blue, alpha=0.7, linewidth=1.5)
                ax1.hist(rs, bins=40, density=True, alpha=0.25, color="white", edgecolor="none")

                # Normal fit
                mu, sigma = stats.norm.fit(rs)
                x = np.linspace(rs.min(), rs.max(), 300)
                y = stats.norm.pdf(x, mu, sigma)
                ax1.plot(x, y, linestyle="--", linewidth=1.8, color="black",
                         label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
                ax1.legend(facecolor=custom_orange, edgecolor="white", labelcolor="white")

                ax1.set_title(f"Density of Daily Returns — {col}", fontsize=13, weight="bold")
                ax1.set_xlabel("Daily Return")
                ax1.set_ylabel("Density")
                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_color("white")
            except Exception as e:
                ax1.text(
                    0.5, 0.5,
                    f"Error plotting density:\n{str(e)}",
                    ha="center", va="center", fontsize=12,
                    color="white", transform=ax1.transAxes
                )
                ax1.set_axis_off()

        # ---------- SAVE + SHOW ----------
        out_file = os.path.join(save_path, f"{col}_graph.png")
        plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

        if show_stock is not None and col == show_stock:
            plt.show()
        else:
            plt.close(fig)

    print(f"Generated {len(stock_df.columns)} plots. Displayed only: {show_stock}")



# -------------------------------------------------------------
# Bivariate Plot
# -------------------------------------------------------------

def save_returns_index_vs_tunindex(df_index, folder_path, tunindex_col="TUNINDEX", show_index=None):
    """
    Plot each index vs TUNINDEX, save all figures, and optionally display one.
    df_index: dataframe containing multiple market indices
    folder_path: where to save PNG plots
    tunindex_col: name of TUNINDEX column in df_index
    show_index: if provided, only this index plot will be shown
    """

    os.makedirs(folder_path, exist_ok=True)

    # Ensure index is datetime
    if not np.issubdtype(df_index.index.dtype, np.datetime64):
        df_index = df_index.copy()
        df_index.index = pd.to_datetime(df_index.index)

    # Ensure TUNINDEX exists
    if tunindex_col not in df_index.columns:
        raise ValueError(f"Column '{tunindex_col}' not found in df_index.")

    tun = df_index[tunindex_col]

    for col in df_index.columns:
        if col == tunindex_col:
            continue  # Avoid plotting TUNINDEX vs itself

        # Prepare fig
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(custom_orange)

        # Grid
        ax.grid(True, color=custom_orange, alpha=0.3)

        # Plot TUNINDEX
        ax.plot(df_index.index, tun, label=tunindex_col,
                linewidth=2, color="black")

        # Plot the other index
        ax.plot(df_index.index, df_index[col], label=col,
                linewidth=2, color=custom_blue)

        # Title & labels
        ax.set_title(f"{col} vs {tunindex_col}", fontsize=16, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Level")
        ax.tick_params(axis="x", rotation=60)

        # Legend
        ax.legend()

        # Save file
        fname = os.path.join(folder_path, f"{col}_vs_{tunindex_col}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=250)
        
        # Show only selected stock
        if show_index is not None and col == show_index:
            plt.show()
        else:
            plt.close()

    print(f"Plots saved in: {folder_path}")


# -------------------------------------------------------------
# Bivariate Plot
# -------------------------------------------------------------

def save_raw_vs_adj_plots(df_raw, df_adj, folder_path):
    """
    Save Raw vs Adjusted Price plots for all columns (stocks) in the given folder.
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Loop through all columns (stocks)
    for stock in df_raw.columns:

        # Create figure with same style as plot_raw_vs_adjusted
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(custom_orange)  # background

        # Plot Raw and Adjusted
        ax.plot(df_raw.index, df_raw[stock], label='Raw Price',
                color=custom_blue, linewidth=2, linestyle='-')
        ax.plot(df_adj.index, df_adj[stock], label='Adjusted Price',
                color=custom_blue, linewidth=2, linestyle='-.')

        # Title and labels
        ax.set_title(f'Raw vs Adjusted Price – {stock}', fontsize=16, weight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Grid and x-axis formatting
        ax.grid(True, color=custom_orange, alpha=0.3)
        ax.tick_params(axis='x', rotation=90)

        # Legend
        ax.legend()

        # Save figure
        file_path = os.path.join(folder_path, f"{stock}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
        plt.close()


# -------------------------------------------------------------
# Bivariate Plot
# -------------------------------------------------------------

def save_index_vs_tunindex(df_index, folder_path, tunindex_col="TUNINDEX", 
                           show_index=None):
    """
    Plot each index vs TUNINDEX using dual y-axes.
    Save all figures, and optionally display one.
    """

    os.makedirs(folder_path, exist_ok=True)

    # Ensure time index
    if not np.issubdtype(df_index.index.dtype, np.datetime64):
        df_index = df_index.copy()
        df_index.index = pd.to_datetime(df_index.index)

    # Check TUNINDEX exists
    if tunindex_col not in df_index.columns:
        raise ValueError(f"Column '{tunindex_col}' not found in df_index.")

    tun = df_index[tunindex_col]

    for col in df_index.columns:
        if col == tunindex_col:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(custom_orange)

        # --- Left Axis (TUNINDEX) ---
        ax1.grid(True, color=custom_orange, alpha=0.3)
        ax1.plot(df_index.index, tun, color="black", linewidth=2, label=tunindex_col)
        ax1.set_ylabel(f"{tunindex_col} Level", color="black", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="black")

        # --- Right Axis (Other Index) ---
        ax2 = ax1.twinx()
        ax2.plot(df_index.index, df_index[col], color=custom_blue, linewidth=2, label=col)
        ax2.set_ylabel(f"{col} Level", color=custom_blue, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=custom_blue)

        # --- Title ---
        ax1.set_title(f"{col} vs {tunindex_col} (Dual Y-Axis)", 
                      fontsize=16, weight="bold")
        ax1.set_xlabel("Date")
        ax1.tick_params(axis="x", rotation=60)

        # --- Save ---
        fname = os.path.join(folder_path, f"{col}_vs_{tunindex_col}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=250, facecolor=fig.get_facecolor())

        # Show only selected index
        if show_index is not None and col == show_index:
            plt.show()
        else:
            plt.close(fig)

    print(f"Plots saved in: {folder_path}")


# -------------------------------------------------------------
# Bivariate Plot
# -------------------------------------------------------------

def save_raw_plots(df_raw, folder_path):
    """
    Save Raw vs Adjusted Price plots for all columns (stocks) in the given folder.
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Loop through all columns (stocks)
    for stock in df_raw.columns:

        # Create figure with same style as plot_raw_vs_adjusted
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(custom_orange)  # background

        # Plot Raw and Adjusted
        ax.plot(df_raw.index, df_raw[stock], label='Raw Price',
                color=custom_blue, linewidth=2, linestyle='-')

        # Title and labels
        ax.set_title(f'Raw vs Adjusted Price – {stock}', fontsize=16, weight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Grid and x-axis formatting
        ax.grid(True, color=custom_orange, alpha=0.3)
        ax.tick_params(axis='x', rotation=90)

        # Legend
        ax.legend()

        # Save figure
        file_path = os.path.join(folder_path, f"{stock}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
        plt.close()
        

