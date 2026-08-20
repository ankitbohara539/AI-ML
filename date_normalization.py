import pandas as pd


def parse_date(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    # Formats to try, in order
    formats = [
        # Year-Month-Day
        "%Y-%m-%d",      # 2025-04-27
        "%Y/%m/%d",      # 2025/04/27
        "%Y.%m.%d",      # 2025.04.27

        # Day-Month-Year
        "%d/%m/%Y",      # 27/04/2025
        "%d-%m-%Y",      # 27-04-2025
        "%d.%m.%Y",      # 27.04.2025

        # Month-Day-Year
        "%m/%d/%Y",      # 04/27/2025
        "%m-%d-%Y",      # 04-27-2025
        "%m.%d.%Y",      # 04.27.2025

        # Written Day-Month-Year
        "%d %B %Y",      # 05 July 2023
        "%d %b %Y",      # 05 Jul 2023
        "%d %B, %Y",     # 05 July, 2023
        "%d %b, %Y",     # 05 Jul, 2023

        # Written Month-Day-Year
        "%B %d, %Y",     # May 07, 2023
        "%B %d %Y",      # May 07 2023
        "%b %d, %Y",     # Apr 27, 2025
        "%b %d %Y",      # Apr 6 2025
    ]

    for fmt in formats:
        date = pd.to_datetime(value, format=fmt, errors="coerce")

        if pd.notna(date):
            return date

    # Try YYYY/DD/MM for values such as 2023-13-05
    date = pd.to_datetime(
        value,
        format="%Y-%d-%m",
        errors="coerce"
    )

    if pd.notna(date):
        return date

    # Nothing worked
    return pd.NaT
    
#------------- Normalization ------------------


print ("\n\nNormalizing date columns\n")