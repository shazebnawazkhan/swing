"""
Data fetching module for the Swing Scanner.

Priority order for delivery + VWAP data:
  1. StockEdge Selenium  (only when USE_STOCKEDGE=True and credentials set)
  2. NSE Archive files   (nsearchives.nseindia.com – no JS cookies required)
  3. NSE JSON API        (nseindia.com – requires JS session, may fail)
  4. yfinance            (last resort – no delivery %, VWAP is approximated)

Priority order for Futures OI data:
  1. NSE FO Archive files (nsearchives.nseindia.com)
  2. NSE JSON API
"""

import os
import time
import logging
import zipfile
from io import BytesIO, StringIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

# Local cache directory for downloaded archive files
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".nse_cache")


def _trading_days(from_date: datetime, to_date: datetime) -> list[datetime]:
    """Return all weekdays in [from_date, to_date] normalised to midnight."""
    days = []
    cur = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = to_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ══════════════════════════════════════════════════════════════════════════════
# NSE ARCHIVE FETCHER  (primary – no cookies needed)
# ══════════════════════════════════════════════════════════════════════════════

class NSEArchiveFetcher:
    """
    Downloads NSE bhav-copy archive CSVs from nsearchives.nseindia.com.
    These plain-text files do not require JS session cookies and are the
    most reliable source for delivery data.

    Equity archive URL:
      https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv

    FO bhav-copy URL:
      https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_DDMMYYYY_F_0000.csv.zip
    """

    # Equity archive: DDMMYYYY  e.g. 22052026
    EQ_URL  = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
    # FO archive: YYYYMMDD  e.g. 20260522
    FO_URL = ("https://nsearchives.nseindia.com/content/fo/"
              "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip")

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_BROWSER_HEADERS)
        os.makedirs(_CACHE_DIR, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cached_get(self, url: str, cache_path: str) -> bytes | None:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return fh.read()
        try:
            resp = self.session.get(url, timeout=20)
        except Exception as exc:
            logger.debug("Archive fetch error: %s", exc)
            return None

        if resp.status_code != 200 or len(resp.content) <= 200:
            logger.debug("Archive HTTP %s: %s", resp.status_code, url)
            return None

        # Separate the cache-write from the network fetch.
        # On Windows, concurrent threads may both try to write the same file;
        # the OSError from the second write must NOT discard the downloaded data.
        try:
            with open(cache_path, "wb") as fh:
                fh.write(resp.content)
        except OSError:
            pass   # another thread already wrote the same file — data is intact

        return resp.content

    # ── equity delivery ───────────────────────────────────────────────────────

    def get_delivery_data(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> pd.DataFrame:
        """
        Build a delivery DataFrame for `symbol` across the date range by
        downloading one bhav-copy file per trading day.
        """
        records = []
        for day in _trading_days(from_date, to_date):
            date_str   = day.strftime("%d%m%Y")
            cache_file = os.path.join(_CACHE_DIR, f"eq_{date_str}.csv")
            url        = self.EQ_URL.format(date=date_str)

            raw = self._cached_get(url, cache_file)
            if raw is None:
                continue

            try:
                text = raw.decode("utf-8", errors="ignore")
                df   = pd.read_csv(StringIO(text))
                df.columns = [c.strip().upper() for c in df.columns]

                # NSE uses different column names across file versions
                sym_col = next(
                    (c for c in df.columns if c in ("SYMBOL", "SYM", "SYMBOL_")), None
                )
                if sym_col is None:
                    continue

                row = df[df[sym_col].str.strip() == symbol]
                if row.empty:
                    continue

                r = row.iloc[0]

                def _f(candidates, default=np.nan):
                    for name in candidates:
                        if name in r.index:
                            try:
                                return float(str(r[name]).replace(",", ""))
                            except (ValueError, TypeError):
                                pass
                    return default

                close    = _f(["CLOSE_PRICE", "CLOSE", "CLOSEPRICE"])
                open_p   = _f(["OPEN_PRICE", "OPEN", "OPENPRICE"])
                high_p   = _f(["HIGH_PRICE", "HIGH", "HIGHPRICE"])
                low_p    = _f(["LOW_PRICE", "LOW", "LOWPRICE"])
                ttl_qty  = _f(["TTL_TRD_QNTY", "TOTTRDQTY", "TOTALTRADEDQUANTITY"])
                del_qty  = _f(["DELIV_QTY", "DELIVERABLEQTY", "DELIVQTY"])
                del_per  = _f(["DELIV_PER", "DELIVERABLE_PER", "DELIVPER"])
                avg_pr   = _f(["AVG_PRICE", "VWAP", "AVGPRICE"])

                # Compute VWAP from turnover if AVG_PRICE not present
                if np.isnan(avg_pr):
                    turnover = _f(["TURNOVER_LACS", "TOTTRDVAL"])
                    if not np.isnan(turnover) and not np.isnan(ttl_qty) and ttl_qty > 0:
                        # TURNOVER_LACS is in lakhs, qty is in units
                        avg_pr = (turnover * 1e5) / ttl_qty
                    else:
                        avg_pr = close  # fallback

                records.append(dict(
                    date         = day,
                    open         = open_p,
                    high         = high_p,
                    low          = low_p,
                    close        = close,
                    total_volume = ttl_qty,
                    delivery_qty = del_qty,
                    delivery_pct = del_per,
                    vwap         = avg_pr,
                ))
            except Exception as exc:
                logger.debug("Parse error for %s on %s: %s", symbol, day.date(), exc)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    # ── futures OI ────────────────────────────────────────────────────────────

    def get_futures_oi(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> pd.DataFrame:
        """
        Build a futures OI DataFrame from FO bhav-copy archives.
        Per-symbol OI results are cached as small CSVs to avoid re-scanning
        the 45k-row full FO files on repeated calls.
        """
        # Per-symbol OI cache: keyed by symbol + date range
        sym_cache = os.path.join(
            _CACHE_DIR,
            f"oi_{symbol}_{from_date:%Y%m%d}_{to_date:%Y%m%d}.csv",
        )
        if os.path.exists(sym_cache):
            df = pd.read_csv(sym_cache)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)

        records = []
        for day in _trading_days(from_date, to_date):
            # FO archive uses YYYYMMDD, equity archive uses DDMMYYYY
            fo_date_str = day.strftime("%Y%m%d")
            cache_file  = os.path.join(_CACHE_DIR, f"fo_{fo_date_str}.zip")
            url         = self.FO_URL.format(date=fo_date_str)

            raw = self._cached_get(url, cache_file)
            if raw is None:
                continue

            try:
                # All FO bhav copies are zipped
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    csv_name = next(
                        (n for n in zf.namelist() if n.endswith(".csv")), None
                    )
                    if csv_name is None:
                        continue
                    text = zf.read(csv_name).decode("utf-8", errors="ignore")

                df = pd.read_csv(StringIO(text))
                df.columns = [c.strip() for c in df.columns]

                # New-format FO bhav copy (2024+): TckrSymb / FinInstrmTp / OpnIntrst
                # Old-format: SYMBOL / INSTRUMENT / OPEN_INT
                sym_col = next(
                    (c for c in df.columns
                     if c.upper() in ("TCKRSYMB", "SYMBOL", "UNDERLYING")),
                    None,
                )
                if sym_col is None:
                    continue

                mask_sym = df[sym_col].astype(str).str.strip() == symbol

                inst_col = next(
                    (c for c in df.columns
                     if c.upper() in ("FININSTRMT P", "FININSTRMTP", "INSTRUMENT",
                                      "INSTTYPE", "INSTRUMENT_TYPE")),
                    None,
                )
                if inst_col:
                    # New format uses 'STF' for stock futures; old uses 'FUTSTK'
                    mask_fut = df[inst_col].astype(str).str.upper().isin(["STF", "FUTSTK"])
                else:
                    mask_fut = pd.Series([True] * len(df), index=df.index)

                rows = df[mask_sym & mask_fut]
                if rows.empty:
                    continue

                def _col_sum(candidates):
                    for name in candidates:
                        # Case-insensitive match
                        match = next(
                            (c for c in rows.columns if c.upper() == name.upper()), None
                        )
                        if match:
                            return pd.to_numeric(
                                rows[match].astype(str).str.replace(",", ""), errors="coerce"
                            ).sum()
                    return np.nan

                oi        = _col_sum(["OpnIntrst", "OPEN_INT", "OPEN_INTEREST", "OPENINT"])
                oi_change = _col_sum(["ChngInOpnIntrst", "CHG_IN_OI", "CHANGE_IN_OI", "CHANGEINOI"])

                if not np.isnan(oi):
                    records.append(dict(date=day, oi=oi, oi_change=oi_change))

            except Exception as exc:
                logger.debug("FO parse error for %s on %s: %s", symbol, day.date(), exc)

        if not records:
            return pd.DataFrame()

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["oi", "oi_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # Save per-symbol cache for fast subsequent access
        df.to_csv(sym_cache, index=False)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# NSE API FETCHER  (secondary – may need JS cookies)
# ══════════════════════════════════════════════════════════════════════════════

class NSEApiFetcher:
    """
    Tries NSE India's JSON historical APIs.
    NSE sets critical cookies via JavaScript, so this may return 503
    in headless environments; it is kept as a secondary fallback.
    """

    BASE = "https://www.nseindia.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            **_BROWSER_HEADERS,
            "Accept":   "application/json, text/plain, */*",
            "Referer":  "https://www.nseindia.com/",
        })
        self._init_session()

    def _init_session(self):
        warm_pages = [
            self.BASE,
            f"{self.BASE}/market-data/live-equity-market",
        ]
        for page in warm_pages:
            try:
                self.session.get(page, timeout=12)
                time.sleep(1.2)
            except Exception:
                pass

    def _get_json(self, url: str) -> dict | None:
        try:
            resp = self.session.get(url, timeout=18)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.debug("NSE API error: %s", exc)
        return None

    def get_delivery_data(self, symbol: str, from_str: str, to_str: str) -> pd.DataFrame:
        url = (
            f"{self.BASE}/api/historical/securityArchives"
            f"?from={from_str}&to={to_str}"
            f"&symbol={symbol}&dataType=priceVolumeDeliverable&series=EQ"
        )
        data = self._get_json(url)
        if not data:
            return pd.DataFrame()

        records = data.get("data", [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        rename = {
            "CH_TIMESTAMP": "date", "CH_OPENING_PRICE": "open",
            "CH_TRADE_HIGH_PRICE": "high", "CH_TRADE_LOW_PRICE": "low",
            "CH_CLOSING_PRICE": "close", "CH_TOT_TRADED_QTY": "total_volume",
            "COP_DELIV_QTY": "delivery_qty", "COP_DELIV_PERC": "delivery_pct",
            "VWAP": "vwap",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        want = ["date", "open", "high", "low", "close",
                "total_volume", "delivery_qty", "delivery_pct", "vwap"]
        df = df[[c for c in want if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_futures_oi(self, symbol: str, from_str: str, to_str: str) -> pd.DataFrame:
        url = (
            f"{self.BASE}/api/historical/fo/derivatives"
            f"?from={from_str}&to={to_str}"
            f"&symbol={symbol}&instrumentType=FUTSTK"
            f"&expiry=&strikePrice=&optionType="
        )
        data = self._get_json(url)
        if not data:
            return pd.DataFrame()
        records = data.get("data") or data.get("filtered", {}).get("data", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        rename = {
            "FH_TIMESTAMP": "date", "FH_OPEN_INT": "oi", "FH_CHANGE_IN_OI": "oi_change",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for col in ["oi", "oi_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        agg = {c: "sum" for c in ["oi", "oi_change"] if c in df.columns}
        df = df.groupby("date").agg(agg).reset_index()
        return df.sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# YFINANCE FALLBACK  (price data only — no delivery %)
# ══════════════════════════════════════════════════════════════════════════════

def _yfinance_fallback(symbol: str, days: int) -> pd.DataFrame:
    """
    Last-resort price fetcher using yfinance.
    delivery_qty / delivery_pct are NaN so strategy condition C1 will be False.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed. pip install yfinance")
        return pd.DataFrame()

    end   = datetime.now()
    start = end - timedelta(days=days + 5)
    try:
        raw = yf.download(
            f"{symbol}.NS",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    # date might be "Date", "Datetime", or first column
    date_col = next(
        (c for c in raw.columns if c.lower() in ("date", "datetime")),
        raw.columns[0],
    )
    raw = raw.rename(columns={date_col: "date", "volume": "total_volume"})
    raw["date"] = pd.to_datetime(raw["date"])
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_localize(None)

    raw["vwap"]         = (raw["high"] + raw["low"] + raw["close"]) / 3
    raw["delivery_qty"] = np.nan
    raw["delivery_pct"] = np.nan

    logger.warning(
        "yfinance data for %s: delivery_qty/pct unavailable — "
        "C1 (delivery up) will always be False.", symbol,
    )
    return raw[["date", "open", "high", "low", "close",
                "total_volume", "delivery_qty", "delivery_pct", "vwap"]].copy()


# ══════════════════════════════════════════════════════════════════════════════
# STOCKEDGE SCRAPER  (Selenium – optional, login required)
# ══════════════════════════════════════════════════════════════════════════════

class StockEdgeScraper:
    """
    Selenium-based scraper for web.stockedge.com.
    Requires: pip install selenium webdriver-manager
    Set USE_STOCKEDGE=True and STOCKEDGE_EMAIL/PASSWORD in config.py.
    """

    LOGIN_URL    = "https://web.stockedge.com/login"
    DELIVERY_URL = "https://web.stockedge.com/share/{slug}/{id}?section=deliveries"
    FUTURES_URL  = "https://web.stockedge.com/share/{slug}/{id}?section=futures"

    def __init__(self, email: str, password: str, headless: bool = True):
        self.email      = email
        self.password   = password
        self.driver     = None
        self._logged_in = False
        self._setup_driver(headless)

    def _setup_driver(self, headless: bool):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            opts = Options()
            if headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--window-size=1280,900")
            opts.add_argument(f"user-agent={_BROWSER_HEADERS['User-Agent']}")
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=opts
            )
        except ImportError:
            logger.error("Install: pip install selenium webdriver-manager")
        except Exception as exc:
            logger.error("ChromeDriver error: %s", exc)

    def login(self) -> bool:
        if not self.driver:
            return False
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            self.driver.get(self.LOGIN_URL)
            wait = WebDriverWait(self.driver, 15)
            email_f = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[type='email'],input[name='email']")
            ))
            email_f.clear(); email_f.send_keys(self.email)
            pwd_f = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            pwd_f.clear(); pwd_f.send_keys(self.password)
            self.driver.find_element(
                By.CSS_SELECTOR, "button[type='submit'],.login-btn,.btn-login"
            ).click()
            wait.until(EC.url_changes(self.LOGIN_URL))
            time.sleep(2)
            self._logged_in = True
            return True
        except Exception as exc:
            logger.error("StockEdge login failed: %s", exc)
            return False

    def _scrape_table(self, url: str) -> pd.DataFrame:
        if not self.driver:
            return pd.DataFrame()
        if not self._logged_in and not self.login():
            return pd.DataFrame()
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from bs4 import BeautifulSoup

            self.driver.get(url)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            time.sleep(1.5)
            soup  = BeautifulSoup(self.driver.page_source, "lxml")
            table = soup.find("table")
            if not table:
                return pd.DataFrame()
            headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
            rows = [
                dict(zip(headers, [td.get_text(strip=True) for td in tr.find_all("td")]))
                for tr in table.find("tbody").find_all("tr")
                if len(tr.find_all("td")) == len(headers)
            ]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as exc:
            logger.warning("StockEdge scrape error: %s", exc)
            return pd.DataFrame()

    def _clean(self, df: pd.DataFrame, col_hints: dict) -> pd.DataFrame:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        rename = {}
        for col in df.columns:
            for canonical, keywords in col_hints.items():
                if any(k in col for k in keywords):
                    rename[col] = canonical
                    break
        df = df.rename(columns=rename)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for col in df.columns:
            if col != "date":
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )
        return df

    def get_delivery_data(self, stockedge_id: int, slug: str) -> pd.DataFrame:
        url = self.DELIVERY_URL.format(slug=slug, id=stockedge_id)
        df  = self._scrape_table(url)
        if df.empty:
            return pd.DataFrame()
        return self._clean(df, {
            "date":         ["date", "time"],
            "delivery_qty": ["del", "qty"],
            "delivery_pct": ["del", "%"],
            "vwap":         ["vwap"],
            "close":        ["close"],
            "total_volume": ["vol", "total"],
        })

    def get_futures_oi(self, stockedge_id: int, slug: str) -> pd.DataFrame:
        url = self.FUTURES_URL.format(slug=slug, id=stockedge_id)
        df  = self._scrape_table(url)
        if df.empty:
            return pd.DataFrame()
        return self._clean(df, {
            "date":      ["date", "time"],
            "oi":        ["open_int", "oi"],
            "oi_change": ["change", "oi"],
        })

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED FETCHER
# ══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Orchestrates all data sources in priority order.
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self.archive  = NSEArchiveFetcher()
        self.nse_api  = NSEApiFetcher()
        self._se: StockEdgeScraper | None = None

        if cfg.USE_STOCKEDGE and cfg.STOCKEDGE_EMAIL and cfg.STOCKEDGE_PASSWORD:
            self._se = StockEdgeScraper(cfg.STOCKEDGE_EMAIL, cfg.STOCKEDGE_PASSWORD)

    @staticmethod
    def _date_window(total_days: int) -> tuple[datetime, datetime, str, str]:
        to_dt  = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        frm_dt = to_dt - timedelta(days=total_days)
        fmt    = "%d-%m-%Y"
        return frm_dt, to_dt, frm_dt.strftime(fmt), to_dt.strftime(fmt)

    def get_stock_data(
        self,
        symbol: str,
        stockedge_id: int,
        stockedge_slug: str,
        has_futures: bool,
        days: int,
    ) -> pd.DataFrame:
        """
        Returns merged DataFrame:
          date, open, high, low, close, total_volume,
          delivery_qty, delivery_pct, vwap, oi, oi_change, symbol
        """
        total_days  = days + self.cfg.DATA_BUFFER_DAYS
        frm_dt, to_dt, frm_str, to_str = self._date_window(total_days)

        # ── Delivery + VWAP ──────────────────────────────────────────────────
        delivery_df = pd.DataFrame()

        if self._se:
            print(f"    [StockEdge] delivery for {symbol}...")
            delivery_df = self._se.get_delivery_data(stockedge_id, stockedge_slug)

        if delivery_df.empty:
            print(f"    [NSE Archive] delivery for {symbol}...")
            delivery_df = self.archive.get_delivery_data(symbol, frm_dt, to_dt)

        if delivery_df.empty:
            print(f"    [NSE API] delivery for {symbol}...")
            delivery_df = self.nse_api.get_delivery_data(symbol, frm_str, to_str)

        if delivery_df.empty:
            print(f"    [yfinance] fallback delivery for {symbol}...")
            delivery_df = _yfinance_fallback(symbol, total_days)

        if delivery_df.empty:
            print(f"    ✗ No data available for {symbol}")
            return pd.DataFrame()

        # ── Futures OI ───────────────────────────────────────────────────────
        oi_df = pd.DataFrame()
        if has_futures:
            if self._se:
                print(f"    [StockEdge] futures OI for {symbol}...")
                oi_df = self._se.get_futures_oi(stockedge_id, stockedge_slug)

            if oi_df.empty:
                print(f"    [NSE Archive] futures OI for {symbol}...")
                oi_df = self.archive.get_futures_oi(symbol, frm_dt, to_dt)

            if oi_df.empty:
                print(f"    [NSE API] futures OI for {symbol}...")
                oi_df = self.nse_api.get_futures_oi(symbol, frm_str, to_str)

        # ── Merge ─────────────────────────────────────────────────────────────
        df = delivery_df.copy()
        if not oi_df.empty:
            merge_cols = ["date"] + [c for c in ["oi", "oi_change"] if c in oi_df.columns]
            df = df.merge(oi_df[merge_cols], on="date", how="left")
        else:
            df["oi"]        = np.nan
            df["oi_change"] = np.nan

        cutoff = datetime.now() - timedelta(days=days)
        df     = df[df["date"] >= cutoff].reset_index(drop=True)
        df.insert(0, "symbol", symbol)
        return df

    def close(self):
        if self._se:
            self._se.close()
