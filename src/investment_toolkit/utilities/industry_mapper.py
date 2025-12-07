"""
industry_mapper.py
─────────────────────────────────────────────────────────────
FMP の "raw" Industry 名を GICS 公式 74 Industry へ変換するユーティリティ
─────────────────────────────────────────────────────────────
使い方：
    import industry_mapper as im
    im.map_gics_industry("Asset Management - Bonds")
        → 'Asset Management & Custody Banks'
"""

from __future__ import annotations
import re

# ------------------------------------------------------------------
# 1. 正規化ヘルパ
# ------------------------------------------------------------------
def _normalize_raw(industry: str | None) -> str:
    """前後スペース / タブ / 改行を除去し、空なら 'Unclassified' を返す"""
    if industry is None:
        return "Unclassified"
    x = re.sub(r"\s+", " ", industry).strip()
    return x if x else "Unclassified"


# ------------------------------------------------------------------
# 2. 正規表現 → GICS Industry 74 本マッピング
#    必要に応じて↓PATTERN_MAPを追記すれば辞書が拡張できる
# ------------------------------------------------------------------
PATTERN_MAP: list[tuple[re.Pattern, str]] = [
    # --- Financials -------------------------------------------------
    (re.compile(r"^Asset Management", re.I), "Asset Management & Custody Banks"),
    (re.compile(r"Investment.*Banking|Capital Markets", re.I), "Investment Banking & Brokerage"),
    (re.compile(r"Financial - Data & Stock Exchanges", re.I), "Financial Exchanges & Data"),
    (re.compile(r"^Banks", re.I), "Banks"),
    (re.compile(r"Financial - Credit Services", re.I), "Consumer Finance"),
    (re.compile(r"Financial - Conglomerates", re.I), "Diversified Financial Services"),
    (re.compile(r"Financial - Mortgages", re.I), "Thrifts & Mortgage Finance"),
    # Insurance
    (re.compile(r"Insurance - Life", re.I), "Life & Health Insurance"),
    (re.compile(r"Insurance.*Diversified", re.I), "Multi-line Insurance"),
    (re.compile(r"Insurance - Property & Casualty", re.I), "Property & Casualty Insurance"),
    (re.compile(r"Insurance - Brokers", re.I), "Insurance Brokers"),
    (re.compile(r"Insurance - Specialty", re.I), "Specialty Insurance"),
    (re.compile(r"Insurance - Reinsurance", re.I), "Reinsurance"),

    # --- Information Technology ------------------------------------
    (re.compile(r"Software", re.I), "Application Software"),  # ※必要に応じてSystems Softwareへ分岐
    (re.compile(r"Internet (Content|Software/Services)", re.I), "Interactive Media & Services"),
    (re.compile(r"Semiconductors?", re.I), "Semiconductors & Semiconductor Equipment"),
    (re.compile(r"Computer Hardware|Technology Distributors|Consumer Electronics", re.I),
     "Technology Hardware, Storage & Peripherals"),
    (re.compile(r"Communication Equipment", re.I), "Communications Equipment"),

    # --- Communication Services ------------------------------------
    (re.compile(r"Telecommunications Services", re.I), "Diversified Telecommunication Services"),
    (re.compile(r"Broadcasting|Entertainment|Electronic Gaming", re.I), "Entertainment"),

    # --- Energy -----------------------------------------------------
    # Energyセクターの業種マッピングを強化
    (re.compile(r"Oil & Gas (Equipment|Services)|Oilfield Services|Drilling", re.I), "Energy Equipment & Services"),
    (re.compile(r"Oil & Gas (Exploration|E&P|Production|Drilling)", re.I), "Oil, Gas & Consumable Fuels"),
    (re.compile(r"Oil & Gas (Integrated|Refining|Midstream)", re.I), "Oil, Gas & Consumable Fuels"),
    (re.compile(r"^(Oil|Gas|Petroleum|Crude)", re.I), "Oil, Gas & Consumable Fuels"),
    (re.compile(r"Uranium|Coal|Natural Gas", re.I), "Oil, Gas & Consumable Fuels"),
    (re.compile(r"Renewable|Solar|Wind|Geothermal|Biofuel", re.I), "Independent Power & Renewable Electricity Producers"),
    (re.compile(r"Utilities -.*Power|Independent Power", re.I), "Independent Power & Renewable Electricity Producers"),
    (re.compile(r"^Energy$", re.I), "Oil, Gas & Consumable Fuels"),  # デフォルトのEnergyはOil & Gasに分類

    # --- Materials --------------------------------------------------
    (re.compile(r"Gold|Silver|Other Precious Metals", re.I), "Metals & Mining"),
    (re.compile(r"Steel|Copper|Aluminum", re.I), "Metals & Mining"),
    (re.compile(r"Chemicals", re.I), "Chemicals"),
    (re.compile(r"Packaging & Containers|Paper, Lumber & Forest Products", re.I),
     "Containers & Packaging"),
    (re.compile(r"Agricultural Inputs|Industrial Materials|Basic Materials", re.I),
     "Materials (Misc)"),

    # --- Industrials -----------------------------------------------
    (re.compile(r"Aerospace & Defense", re.I), "Aerospace & Defense"),
    (re.compile(r"Industrial - Machinery|Specialty Industrial Machinery|Tools & Accessories|Capital Goods",
               re.I), "Industrial Machinery"),
    (re.compile(r"Integrated Freight|Roads? & Rail|Trucking|Marine Shipping|Railroads", re.I),
     "Air Freight & Logistics"),
    (re.compile(r"Construction Materials|Engineering & Construction|Construction$", re.I),
     "Construction & Engineering"),

    # --- Consumer Discretionary ------------------------------------
    (re.compile(r"Apparel|Luxury Goods|Footwear", re.I), "Textiles, Apparel & Luxury Goods"),
    (re.compile(r"Specialty Retail|Department Stores|Home Improvement|Discount Stores", re.I),
     "Specialty Retail"),
    (re.compile(r"Restaurants|Hotels|Leisure|Casinos|Travel", re.I), "Hotels, Restaurants & Leisure"),
    (re.compile(r"Auto - ", re.I), "Automobiles"),
    (re.compile(r"Consumer Electronics", re.I), "Household Durables"),

    # --- Consumer Staples ------------------------------------------
    (re.compile(r"Grocery Stores|Packaged Foods|Food Distribution|Beverages|Household & Personal Products",
               re.I), "Food & Staples Retailing"),
    (re.compile(r"Tobacco", re.I), "Tobacco"),

    # --- Health Care -----------------------------------------------
    (re.compile(r"Pharmaceutical", re.I), "Pharmaceuticals"),
    (re.compile(r"Biotechnology", re.I), "Biotechnology"),
    (re.compile(r"Medical - Diagnostics|Life Sciences Tools", re.I), "Life Sciences Tools & Services"),
    (re.compile(r"Medical - (Care Facilities|Distribution|Healthcare Plans|Nursing Services)",
               re.I), "Health Care Providers & Services"),
    (re.compile(r"Medical - Devices|Instruments|Equipment & Supplies", re.I),
     "Health Care Equipment & Supplies"),

    # --- Real Estate -----------------------------------------------
    (re.compile(r"REIT - Mortgage", re.I), "Mortgage REITs"),
    (re.compile(r"REIT", re.I), "Equity REITs"),
    (re.compile(r"Real Estate - ", re.I), "Real Estate Management & Development"),
]

# ------------------------------------------------------------------
# 3. 変換関数（公開 API）
# ------------------------------------------------------------------
def map_gics_industry(raw: str | None) -> str:
    """
    raw industry 文字列 → GICS Industry 名（74本）へ変換。
    PATTERN_MAP にヒットしなければ 'Unclassified' を返す
    """
    norm = _normalize_raw(raw)
    for pattern, gics in PATTERN_MAP:
        if pattern.search(norm):
            return gics
    return "Unclassified"


# ------------------------------------------------------------------
# 4. デフォルトセクターのマッピング
# ------------------------------------------------------------------
INDUSTRY_TO_SECTOR_MAP = {
    # Financials
    "Asset Management & Custody Banks": "Financials",
    "Investment Banking & Brokerage": "Financials",
    "Financial Exchanges & Data": "Financials",
    "Banks": "Financials",
    "Consumer Finance": "Financials",
    "Diversified Financial Services": "Financials",
    "Thrifts & Mortgage Finance": "Financials",
    "Life & Health Insurance": "Financials",
    "Multi-line Insurance": "Financials",
    "Property & Casualty Insurance": "Financials",
    "Insurance Brokers": "Financials",
    "Specialty Insurance": "Financials",
    "Reinsurance": "Financials",
    
    # Information Technology
    "Application Software": "Information Technology",
    "Interactive Media & Services": "Information Technology",
    "Semiconductors & Semiconductor Equipment": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    "Communications Equipment": "Information Technology",
    
    # Communication Services
    "Diversified Telecommunication Services": "Communication Services",
    "Entertainment": "Communication Services",
    
    # Energy
    "Energy Equipment & Services": "Energy",
    "Oil, Gas & Consumable Fuels": "Energy",
    "Independent Power & Renewable Electricity Producers": "Energy",
    
    # Materials
    "Metals & Mining": "Materials",
    "Chemicals": "Materials",
    "Containers & Packaging": "Materials",
    "Materials (Misc)": "Materials",
    
    # Industrials
    "Aerospace & Defense": "Industrials",
    "Industrial Machinery": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Construction & Engineering": "Industrials",
    
    # Consumer Discretionary
    "Textiles, Apparel & Luxury Goods": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Hotels, Restaurants & Leisure": "Consumer Discretionary",
    "Automobiles": "Consumer Discretionary",
    "Household Durables": "Consumer Discretionary",
    
    # Consumer Staples
    "Food & Staples Retailing": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    
    # Health Care
    "Pharmaceuticals": "Health Care",
    "Biotechnology": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Health Care Providers & Services": "Health Care",
    "Health Care Equipment & Supplies": "Health Care",
    
    # Real Estate
    "Mortgage REITs": "Real Estate",
    "Equity REITs": "Real Estate",
    "Real Estate Management & Development": "Real Estate",
    
    # 未分類
    "Unclassified": "Unclassified"
}

def map_gics_sector(industry: str | None) -> str:
    """
    Industry名からセクター名へのマッピング
    先にmap_gics_industryでIndustryを正規化してから使うことを推奨
    """
    if industry is None:
        return "Unclassified"
    
    gics_industry = map_gics_industry(industry)
    return INDUSTRY_TO_SECTOR_MAP.get(gics_industry, "Unclassified")


# ------------------------------------------------------------------
# 5. FMP raw Sector → GICS Sector (11本) 変換マップ
# ------------------------------------------------------------------
SECTOR_MAP = {
    # 同義語変換
    "Financial Services": "Financials",
    
    # Consumer系
    "Consumer Defensive": "Consumer Staples",
    "Consumer Cyclical": "Consumer Discretionary",
    
    # IT系
    "Technology": "Information Technology",
    
    # Energy系（明示的にマッピング）
    "Energy": "Energy",
    
    # その他の明示的マッピング
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

def map_raw_sector(raw_sector: str | None) -> str:
    """
    FMPの生のセクター文字列をGICS標準セクター(11本)に変換
    
    Parameters:
    -----------
    raw_sector : str | None
        FMPから取得した元のセクター文字列
    
    Returns:
    --------
    str
        GICS標準セクター名
    """
    if raw_sector is None:
        return "Unclassified"
    
    # 前後の空白を除去して正規化
    clean_sector = raw_sector.strip() if isinstance(raw_sector, str) else raw_sector
    
    # マッピング辞書から変換後の値を取得（存在しなければUnclassified）
    return SECTOR_MAP.get(clean_sector, "Unclassified")


# ------------------------------------------------------------------
# 6. データベース変換用関数
# ------------------------------------------------------------------
def normalize_company_industry_data(raw_industry: str | None, raw_sector: str | None = None) -> tuple[str, str]:
    """
    FMPから取得した生のindustry/sectorデータを正規化して返す
    
    Parameters:
    -----------
    raw_industry : str | None
        FMPから取得した元のindustry文字列
    raw_sector : str | None
        FMPから取得した元のsector文字列（指定なしの場合はindustryから推定）
    
    Returns:
    --------
    tuple[str, str]
        (正規化されたGICS Industry, 正規化されたGICSセクター)
    """
    normalized_industry = map_gics_industry(raw_industry)
    
    # セクターが指定されていない場合はindustryから推定
    if raw_sector is None or raw_sector.strip() == "":
        normalized_sector = INDUSTRY_TO_SECTOR_MAP.get(normalized_industry, "Unclassified")
    else:
        # セクターが指定されている場合はGICSセクターにマッピング
        normalized_sector = map_raw_sector(raw_sector)
    
    return (normalized_industry, normalized_sector)


# ------------------------------------------------------------------
# 7. 簡易テスト（直接実行した場合のみ走る）
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_industries = [
        "Asset Management - Bonds",
        "Oil & Gas Equipment & Services",
        "Pharmaceutical",
        "REIT - Office",
        None,
        "",
        "Industrial - Machinery",
    ]
    
    print("Industry Mapping Test:")
    print("-" * 70)
    for industry in test_industries:
        mapped_industry = map_gics_industry(industry)
        mapped_sector = map_gics_sector(industry)
        print(f"{industry!r:40s} → {mapped_industry:40s} → {mapped_sector}")
    
    print("\nRaw Sector Mapping Test:")
    print("-" * 70)
    test_sectors = [
        "Financial Services",
        "Consumer Defensive",
        "Consumer Cyclical",
        "Technology",
        "Basic Materials",
        "Industrials", 
        "Utilities",
        "Real Estate",
        "Communication Services",
        "Energy",
        "Healthcare",
        "",
        None
    ]
    
    for sector in test_sectors:
        print(f"{sector!r:25s} → {map_raw_sector(sector)}")
    
    print("\nNormalize Company Data Test:")
    print("-" * 70)
    test_cases = [
        ("Asset Management - Bonds", "Financial Services"),
        ("Oil & Gas Equipment & Services", "Energy"),
        ("Pharmaceutical", "Healthcare"),
        ("REIT - Office", None),
        (None, "Technology"),
        ("", ""),
    ]
    
    for raw_industry, raw_sector in test_cases:
        norm_industry, norm_sector = normalize_company_industry_data(raw_industry, raw_sector)
        print(f"Raw: ({raw_industry!r}, {raw_sector!r}) → Normalized: ({norm_industry}, {norm_sector})") 