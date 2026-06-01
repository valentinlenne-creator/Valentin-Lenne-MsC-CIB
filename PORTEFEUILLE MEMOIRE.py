import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import par84
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

images = par84.img_path + 'marko-'

# ============================================================
# PARAMETERS
# ============================================================
LAMBDA_CONCENTRATION  = 2.5
FUNDAMENTALS_CACHE    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'fundamentals_cache.csv'
)
CACHE_VALIDITY_HOURS  = 24

# ============================================================
# SECTOR MAPPING
# ============================================================
SECTOR_MAP = {
    'AAPL':'Tech_USA','MSFT':'Tech_USA','GOOGL':'Tech_USA',
    'META':'Tech_USA','AMZN':'Tech_USA','TSLA':'Tech_USA',
    'CRM':'Tech_USA','ORCL':'Tech_USA','ADBE':'Tech_USA',
    'NOW':'Tech_USA','PANW':'Tech_USA','CRWD':'Tech_USA',
    'ZS':'Tech_USA','NET':'Tech_USA','DDOG':'Tech_USA',
    'SNOW':'Tech_USA','PLTR':'Tech_USA','FTNT':'Tech_USA',
    'UBER':'Tech_USA','ABNB':'Tech_USA','DASH':'Tech_USA',
    'COIN':'Tech_USA','TWLO':'Tech_USA','SPOT':'Tech_USA',
    'NFLX':'Tech_USA','DIS':'Tech_USA','WBD':'Tech_USA',
    'CMCSA':'Tech_USA',
    'NVDA':'Semis','AMD':'Semis','INTC':'Semis',
    'QCOM':'Semis','TXN':'Semis','AMAT':'Semis',
    'MU':'Semis','LRCX':'Semis','KLAC':'Semis',
    'SNPS':'Semis','CDNS':'Semis','MRVL':'Semis',
    'AVGO':'Semis','TSM':'Semis','ASX':'Semis',
    'UMC':'Semis','HIMX':'Semis','SIMO':'Semis',
    'JPM':'Finance_USA','BAC':'Finance_USA','WFC':'Finance_USA',
    'GS':'Finance_USA','MS':'Finance_USA','BLK':'Finance_USA',
    'C':'Finance_USA','AXP':'Finance_USA','V':'Finance_USA',
    'MA':'Finance_USA','PYPL':'Finance_USA','COF':'Finance_USA',
    'USB':'Finance_USA','PNC':'Finance_USA','TFC':'Finance_USA',
    'SCHW':'Finance_USA','ICE':'Finance_USA','CME':'Finance_USA',
    'SPGI':'Finance_USA','MCO':'Finance_USA','FIS':'Finance_USA',
    'FISV':'Finance_USA','ALLY':'Finance_USA','HOOD':'Finance_USA',
    'SOFI':'Finance_USA','NDAQ':'Finance_USA','BX':'Finance_USA',
    'KKR':'Finance_USA','APO':'Finance_USA',
    'BNP.PA':'Finance_EU','SAN.MC':'Finance_EU',
    'INGA.AS':'Finance_EU','DBK.DE':'Finance_EU',
    'CABK.MC':'Finance_EU','UCG.MI':'Finance_EU',
    'ISP.MI':'Finance_EU','CS.PA':'Finance_EU',
    'HSBA.L':'Finance_EU','LLOY.L':'Finance_EU',
    'BARC.L':'Finance_EU','NWG.L':'Finance_EU',
    'PRU.L':'Finance_EU','AV.L':'Finance_EU',
    'LGEN.L':'Finance_EU','ALV.DE':'Finance_EU',
    'MUV2.DE':'Finance_EU','ZURN.SW':'Finance_EU',
    'JNJ':'Health','UNH':'Health','PFE':'Health',
    'ABBV':'Health','MRK':'Health','TMO':'Health',
    'ABT':'Health','LLY':'Health','DHR':'Health',
    'BMY':'Health','AMGN':'Health','GILD':'Health',
    'REGN':'Health','VRTX':'Health','ISRG':'Health',
    'SYK':'Health','BSX':'Health','MDT':'Health',
    'EW':'Health','ZBH':'Health','IDXX':'Health',
    'IQV':'Health','A':'Health','WAT':'Health',
    'DXCM':'Health','PODD':'Health','BIO':'Health',
    'MTD':'Health','NOVN.SW':'Health','ROG.SW':'Health',
    'NOVO-B.CO':'Health','AZN.L':'Health','GSK.L':'Health',
    'SAN.PA':'Health','UCB.BR':'Health','FRE.DE':'Health',
    'GE':'Industrials','HON':'Industrials','MMM':'Industrials',
    'CAT':'Industrials','DE':'Industrials','LMT':'Industrials',
    'RTX':'Industrials','NOC':'Industrials','GD':'Industrials',
    'BA':'Industrials','UPS':'Industrials','FDX':'Industrials',
    'CSX':'Industrials','UNP':'Industrials','NSC':'Industrials',
    'EMR':'Industrials','ETN':'Industrials','PH':'Industrials',
    'ROK':'Industrials','IR':'Industrials','AME':'Industrials',
    'CARR':'Industrials','OTIS':'Industrials','TT':'Industrials',
    'GWW':'Industrials','FAST':'Industrials','SNA':'Industrials',
    'HWM':'Industrials','TDG':'Industrials','AIR.PA':'Industrials',
    'SIE.DE':'Industrials','ABB.ST':'Industrials','RR.L':'Industrials',
    'BA.L':'Industrials','LDO.MI':'Industrials','SAF.PA':'Industrials',
    'MTX.DE':'Industrials','SAND.ST':'Industrials',
    'ALFA.ST':'Industrials','VOLV-B.ST':'Industrials',
    'ASSA-B.ST':'Industrials','DSV.CO':'Industrials',
    'MCD':'Consumer','SBUX':'Consumer','NKE':'Consumer',
    'TGT':'Consumer','WMT':'Consumer','COST':'Consumer',
    'HD':'Consumer','LOW':'Consumer','TJX':'Consumer',
    'ROST':'Consumer','DG':'Consumer','DLTR':'Consumer',
    'YUM':'Consumer','DPZ':'Consumer','CMG':'Consumer',
    'LULU':'Consumer','PG':'Consumer','KO':'Consumer',
    'PEP':'Consumer','CL':'Consumer','KMB':'Consumer',
    'GIS':'Consumer','MKC':'Consumer','HSY':'Consumer',
    'MNST':'Consumer','EL':'Consumer','CLX':'Consumer',
    'CHD':'Consumer','STZ':'Consumer','MC.PA':'Consumer',
    'OR.PA':'Consumer','CFR.SW':'Consumer','RMS.PA':'Consumer',
    'KER.PA':'Consumer','RACE.MI':'Consumer','EL.PA':'Consumer',
    'BN.PA':'Consumer','NESN.SW':'Consumer','ULVR.L':'Consumer',
    'ABF.L':'Consumer','TSCO.L':'Consumer','HEIA.AS':'Consumer',
    'XOM':'Energy','CVX':'Energy','COP':'Energy',
    'EOG':'Energy','SLB':'Energy','PSX':'Energy',
    'VLO':'Energy','MPC':'Energy','OXY':'Energy',
    'HAL':'Energy','DVN':'Energy','BKR':'Energy',
    'CF':'Energy','TTE.PA':'Energy','SHEL.L':'Energy',
    'BP.L':'Energy','ENI.MI':'Energy','REP.MC':'Energy',
    'EQNR.OL':'Energy','OMV.VI':'Energy',
    'ASML.AS':'Tech_EU','SAP.DE':'Tech_EU','CAP.PA':'Tech_EU',
    'NOKIA.HE':'Tech_EU','ADYEN.AS':'Tech_EU',
    'TELIA.ST':'Tech_EU','KPN.AS':'Tech_EU',
    'DTE.DE':'Tech_EU','TEF.MC':'Tech_EU',
    'VOD.L':'Tech_EU','BT-A.L':'Tech_EU',
    'VZ':'Tech_EU','T':'Tech_EU','TMUS':'Tech_EU','CHTR':'Tech_EU',
    'TCEHY':'Asia_Tech','BIDU':'Asia_Tech','BABA':'Asia_Tech',
    'JD':'Asia_Tech','PDD':'Asia_Tech','NTES':'Asia_Tech',
    'BEKE':'Asia_Tech','LI':'Asia_Tech','NIO':'Asia_Tech',
    'XPEV':'Asia_Tech','FUTU':'Asia_Tech','VNET':'Asia_Tech',
    'GDS':'Asia_Tech','GRAB':'Asia_Tech','SEA':'Asia_Tech',
    'SONY':'Asia_Tech','NTDOY':'Asia_Tech','FANUY':'Asia_Tech',
    'KYOCY':'Asia_Tech','FUJIY':'Asia_Tech','HOCPY':'Asia_Tech',
    'DSNKY':'Asia_Tech','ITOCY':'Asia_Tech',
    'MFG':'Asia_Finance','MUFG':'Asia_Finance',
    'SMFG':'Asia_Finance','NMR':'Asia_Finance',
    'TM':'Asia_Finance','HMC':'Asia_Finance',
    'HDB':'Asia_Finance','IBN':'Asia_Finance',
    'INFY':'Asia_Finance','WIT':'Asia_Finance',
    'CTSH':'Asia_Finance','PKX':'Asia_Finance',
    'KB':'Asia_Finance','SHG':'Asia_Finance',
    'WF':'Asia_Finance','ANZ.AX':'Asia_Finance',
    'CBA.AX':'Asia_Finance','WBC.AX':'Asia_Finance',
    'NAB.AX':'Asia_Finance','MQG.AX':'Asia_Finance',
    'CSL.AX':'Asia_Finance',
    'LIN':'Materials','APD':'Materials','ECL':'Materials',
    'SHW':'Materials','NEM':'Materials','FCX':'Materials',
    'NUE':'Materials','VMC':'Materials','MLM':'Materials',
    'IFF':'Materials','PPG':'Materials','GLEN.L':'Materials',
    'RIO.L':'Materials','AAL.L':'Materials','BHP.L':'Materials',
    'ANTO.L':'Materials','SIKA.SW':'Materials',
    'BHP':'Materials','RIO':'Materials',
    'FMG.AX':'Materials','WES.AX':'Materials','WOW.AX':'Materials',
    'NEE':'Utils_RE','DUK':'Utils_RE','SO':'Utils_RE',
    'D':'Utils_RE','AEP':'Utils_RE','EXC':'Utils_RE',
    'XEL':'Utils_RE','WEC':'Utils_RE','ES':'Utils_RE',
    'AWK':'Utils_RE','CMS':'Utils_RE','AMT':'Utils_RE',
    'PLD':'Utils_RE','CCI':'Utils_RE','EQIX':'Utils_RE',
    'SPG':'Utils_RE','O':'Utils_RE','DLR':'Utils_RE',
    'PSA':'Utils_RE','WELL':'Utils_RE','VTR':'Utils_RE',
    'EXR':'Utils_RE','AVB':'Utils_RE','EQR':'Utils_RE',
    'ENEL.MI':'Utils_RE','IBE.MC':'Utils_RE',
    'EDP.LS':'Utils_RE','RWE.DE':'Utils_RE',
    'FORTUM.HE':'Utils_RE','ANA.MC':'Utils_RE',
    'ENGI.PA':'Utils_RE','URW.PA':'Utils_RE',
    'LEG.DE':'Utils_RE','VNA.DE':'Utils_RE',
}

SECTOR_QUOTAS = {
    'Tech_USA'    : 4,
    'Semis'       : 3,
    'Finance_USA' : 2,
    'Finance_EU'  : 2,
    'Health'      : 2,
    'Industrials' : 3,
    'Consumer'    : 2,
    'Energy'      : 1,
    'Tech_EU'     : 2,
    'Asia_Tech'   : 1,
    'Asia_Finance': 1,
    'Materials'   : 1,
    'Utils_RE'    : 1,
}

# ============================================================
# INVESTMENT UNIVERSE
# ============================================================
symbols = [
    'AAPL','MSFT','GOOGL','NVDA','META','AMZN','TSLA','AMD',
    'INTC','CRM','ORCL','ADBE','QCOM','TXN','AMAT','MU',
    'LRCX','KLAC','SNPS','CDNS','MRVL','AVGO','NOW','PANW',
    'CRWD','ZS','NET','DDOG','SNOW','PLTR','FTNT',
    'UBER','ABNB','DASH','COIN','TWLO',
    'JPM','BAC','WFC','GS','MS','BLK','C','AXP',
    'V','MA','PYPL','COF','USB','PNC','TFC','SCHW',
    'ICE','CME','SPGI','MCO','FIS','FISV','ALLY',
    'HOOD','SOFI','NDAQ','BX','KKR','APO',
    'JNJ','UNH','PFE','ABBV','MRK','TMO','ABT','LLY',
    'DHR','BMY','AMGN','GILD','REGN','VRTX','ISRG','SYK',
    'BSX','MDT','EW','ZBH','IDXX','IQV','A','WAT',
    'DXCM','PODD','BIO','MTD',
    'MCD','SBUX','NKE','TGT','WMT','COST','HD','LOW',
    'TJX','ROST','DG','DLTR','YUM','DPZ','CMG','LULU',
    'PG','KO','PEP','CL','KMB','GIS','MKC','HSY',
    'MNST','EL','CLX','CHD','STZ',
    'XOM','CVX','COP','EOG','SLB','PSX','VLO','MPC',
    'OXY','HAL','DVN','BKR','CF',
    'GE','HON','MMM','CAT','DE','LMT','RTX','NOC',
    'GD','BA','UPS','FDX','CSX','UNP','NSC','EMR',
    'ETN','PH','ROK','IR','AME','CARR','OTIS','TT',
    'GWW','FAST','SNA','HWM','TDG',
    'AMT','PLD','CCI','EQIX','SPG','O','DLR','PSA',
    'WELL','VTR','EXR','AVB','EQR',
    'NEE','DUK','SO','D','AEP','EXC','XEL','WEC',
    'ES','AWK','CMS',
    'LIN','APD','ECL','SHW','NEM','FCX','NUE','VMC',
    'MLM','IFF','PPG',
    'T','VZ','TMUS','CHTR',
    'DIS','NFLX','CMCSA','WBD','SPOT',
    'BNP.PA','SAN.MC','INGA.AS','DBK.DE','CABK.MC',
    'UCG.MI','ISP.MI','CS.PA','HSBA.L','LLOY.L',
    'BARC.L','NWG.L','PRU.L','AV.L','LGEN.L',
    'ALV.DE','MUV2.DE','ZURN.SW',
    'NOVN.SW','ROG.SW','NOVO-B.CO','AZN.L','GSK.L',
    'SAN.PA','UCB.BR','FRE.DE',
    'ASML.AS','SAP.DE','CAP.PA','NOKIA.HE','ADYEN.AS',
    'AIR.PA','SIE.DE','ABB.ST','RR.L','BA.L',
    'LDO.MI','SAF.PA','MTX.DE','SAND.ST','ALFA.ST',
    'VOLV-B.ST','ASSA-B.ST','DSV.CO',
    'MC.PA','OR.PA','CFR.SW','RMS.PA','KER.PA',
    'RACE.MI','EL.PA','BN.PA','NESN.SW','ULVR.L',
    'ABF.L','TSCO.L','HEIA.AS',
    'TTE.PA','SHEL.L','BP.L','ENI.MI','REP.MC',
    'EQNR.OL','OMV.VI',
    'DTE.DE','TEF.MC','VOD.L','BT-A.L','TELIA.ST','KPN.AS',
    'ENEL.MI','IBE.MC','EDP.LS','RWE.DE','FORTUM.HE',
    'ANA.MC','ENGI.PA',
    'GLEN.L','RIO.L','AAL.L','BHP.L','ANTO.L','SIKA.SW',
    'URW.PA','LEG.DE','VNA.DE',
    'TM','SONY','NTDOY','HMC','FANUY','KYOCY',
    'MFG','MUFG','SMFG','NMR','ITOCY','DSNKY','FUJIY','HOCPY',
    'TCEHY','BIDU','BABA','JD','PDD','NTES',
    'BEKE','LI','NIO','XPEV','FUTU','VNET','GDS',
    'HDB','IBN','INFY','WIT','CTSH',
    'TSM','UMC','ASX','HIMX','SIMO',
    'GRAB','SEA',
    'BHP','RIO','ANZ.AX','CBA.AX','WBC.AX',
    'NAB.AX','CSL.AX','WES.AX','WOW.AX','FMG.AX','MQG.AX',
    'PKX','KB','SHG','WF',
]

print(f"Total universe: {len(symbols)} companies")

# ============================================================
# PRICE DOWNLOAD BY BATCH
# ============================================================
print("\n==== Downloading price data ====")
batch_size = 50
all_data   = []

for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i + batch_size]
    print(f"Batch {i // batch_size + 1}: {batch[0]} → {batch[-1]}")
    try:
        batch_df = yf.Tickers(batch).history(
            period="3y", auto_adjust=True)['Close']
        batch_df = batch_df.tz_localize(None)
        all_data.append(batch_df)
    except Exception as e:
        print(f"  Error: {e}")

df_raw = pd.concat(all_data, axis=1)
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
print(f"Raw data: {df_raw.shape[0]} days x {df_raw.shape[1]} assets")

threshold     = int(0.95 * len(df_raw))
df            = df_raw.dropna(axis=1, thresh=threshold)
df            = df.ffill().dropna()
symbols_clean = list(df.columns)
print(f"Assets after cleaning (95%): {len(symbols_clean)}")

log_returns   = np.log(df / df.shift(1)).dropna()
daily_returns = df.pct_change().dropna()

print("\n==== Risk-free rate ====")
try:
    rf = yf.Ticker('^IRX').history(
        period='1d')['Close'].dropna().iloc[-1] / 100
except:
    rf = 0.04
print(f"Risk-free rate: {rf:.4%}")

# ============================================================
# BENCHMARKS
# ============================================================
try:
    sp500_raw  = yf.Ticker('^GSPC').history(period='3y')['Close']
    sp500_raw.index = sp500_raw.index.tz_localize(None)
    market_raw = sp500_raw.pct_change().dropna()
except:
    sp500_raw  = None
    market_raw = None

try:
    stoxx_raw = yf.Ticker('^STOXX50E').history(period='3y')['Close']
    stoxx_raw.index = stoxx_raw.index.tz_localize(None)
except:
    stoxx_raw = None

try:
    nikkei_raw = yf.Ticker('^N225').history(period='3y')['Close']
    nikkei_raw.index = nikkei_raw.index.tz_localize(None)
except:
    nikkei_raw = None

# ============================================================
# EWMA + LEDOIT-WOLF — SHARED ACROSS ALL 3 PORTFOLIOS
# ============================================================
halflife = 252
ewma_w   = np.array([
    (0.5 ** (1/halflife)) ** i
    for i in range(len(log_returns) - 1, -1, -1)
])
ewma_w  /= ewma_w.sum()

mean_ewma = pd.Series({
    col: np.dot(ewma_w, log_returns[col]) * 252
    for col in log_returns.columns
})

lw            = LedoitWolf()
lw.fit(log_returns)
cov_lw_matrix = pd.DataFrame(
    lw.covariance_ * 252,
    index=log_returns.columns,
    columns=log_returns.columns
)
print(f"Ledoit-Wolf shrinkage: {lw.shrinkage_:.4f}")

# ============================================================
# FUNDAMENTALS COLLECTION — MULTITHREADING + CACHE
# ============================================================
def fetch_fund_single(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'ticker'          : ticker,
            'pe_ratio'        : info.get('trailingPE',         np.nan),
            'pb_ratio'        : info.get('priceToBook',        np.nan),
            'ev_ebitda'       : info.get('enterpriseToEbitda', np.nan),
            'roe'             : info.get('returnOnEquity',     np.nan),
            'operating_margin': info.get('operatingMargins',   np.nan),
        }
    except:
        return {
            'ticker': ticker, 'pe_ratio': np.nan, 'pb_ratio': np.nan,
            'ev_ebitda': np.nan, 'roe': np.nan, 'operating_margin': np.nan
        }

def load_fundamentals(tickers, cache_path, cache_h=24, max_w=20):
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age < cache_h:
            print(f"  Valid cache ({age:.1f}h)")
            df_c    = pd.read_csv(cache_path, index_col='ticker')
            missing = [t for t in tickers if t not in df_c.index]
            if missing:
                new_rows = []
                with ThreadPoolExecutor(max_workers=max_w) as ex:
                    for f in as_completed(
                        {ex.submit(fetch_fund_single, t): t for t in missing}
                    ):
                        new_rows.append(f.result())
                df_c = pd.concat([
                    df_c, pd.DataFrame(new_rows).set_index('ticker')
                ])
                df_c.to_csv(cache_path)
            return df_c.loc[[t for t in tickers if t in df_c.index]]

    print(f"  Downloading ({len(tickers)} assets, {max_w} threads)...")
    t0, results = time.time(), []
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(fetch_fund_single, t): t for t in tickers}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tickers)} — {time.time()-t0:.0f}s")
    print(f"  Completed in {time.time()-t0:.1f}s")
    df_f = pd.DataFrame(results).set_index('ticker')
    df_f.to_csv(cache_path)
    return df_f

print("\n==== Collecting fundamental data ====")
fund_df = load_fundamentals(
    symbols_clean, FUNDAMENTALS_CACHE, CACHE_VALIDITY_HOURS
)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def detect_zone(ticker):
    eu = ['.PA','.DE','.MI','.MC','.L','.AS','.SW',
          '.ST','.CO','.HE','.BR','.VI','.LS','.OL','.WA']
    ax = ['.AX','.T','.HK','.KS','.TW']
    asia_us = [
        'TM','SONY','NTDOY','HMC','FANUY','KYOCY','MFG','MUFG',
        'SMFG','NMR','ITOCY','DSNKY','FUJIY','HOCPY','TCEHY',
        'BIDU','BABA','JD','PDD','NTES','BEKE','LI','NIO','XPEV',
        'FUTU','VNET','GDS','HDB','IBN','INFY','WIT','TSM','UMC',
        'ASX','HIMX','SIMO','GRAB','SEA','BHP','RIO','PKX','KB',
        'SHG','WF','CTSH'
    ]
    if any(ticker.endswith(s) for s in eu):  return 'Europe'
    if any(ticker.endswith(s) for s in ax):  return 'Asia'
    if ticker in asia_us:                     return 'Asia'
    return 'USA'

def normalize(s):
    mn, mx = s.min(), s.max()
    return s * 0 if mx == mn else (s - mn) / (mx - mn)

# ============================================================
# BASE METRICS COMPUTATION
# ============================================================
print("\n==== Computing metrics ====")

daily_ret_1y = daily_returns.tail(252)
daily_ret_6m = daily_returns.tail(126)
base_data    = []

for ticker in symbols_clean:
    try:
        r3  = daily_returns[ticker].dropna()
        r1  = daily_ret_1y[ticker].dropna()
        r6  = daily_ret_6m[ticker].dropna()

        ar3 = r3.mean() * 252
        av3 = r3.std()  * np.sqrt(252)
        ar1 = r1.mean() * 252
        av1 = r1.std()  * np.sqrt(252)
        ar6 = r6.mean() * 252
        av6 = r6.std()  * np.sqrt(252)

        sh3 = (ar3 - rf) / av3 if av3 > 0 else 0
        sh1 = (ar1 - rf) / av1 if av1 > 0 else 0
        sh6 = (ar6 - rf) / av6 if av6 > 0 else 0
        sh_combined = 0.40*sh1 + 0.30*sh6 + 0.30*sh3

        beta, alpha = 1.0, 0.0
        if market_raw is not None:
            common = r3.index.intersection(market_raw.index)
            if len(common) > 100:
                r     = r3.loc[common]
                m     = market_raw.loc[common]
                beta  = r.cov(m) / m.var()
                alpha = ar3 - rf - beta * (m.mean() * 252 - rf)

        ret_std = r3.std()
        z_ret   = (r3.tail(30).mean() - r3.mean()) / ret_std \
                  if ret_std > 0 else 0

        roll30  = r3.rolling(30).apply(
            lambda x: (1+x).prod()-1, raw=True).dropna()
        mom_c   = roll30.iloc[-1] if len(roll30) > 0 else 0
        z_mom   = (mom_c - roll30.mean()) / roll30.std() \
                  if roll30.std() > 0 else 0

        rv30    = r3.rolling(30).std().dropna()
        z_vol   = -((rv30.iloc[-1] - rv30.mean()) / rv30.std()) \
                  if rv30.std() > 0 else 0

        bd      = abs(beta - 1.0)
        z_beta  = (1.0 if bd <= 0.5
                   else (1.0 - (bd - 0.5)) if bd <= 1.0
                   else (0.5 - (bd - 1.0) * 0.5)) - 0.5

        z_final = 0.35*z_ret + 0.35*z_mom + 0.20*z_vol + 0.10*z_beta

        cumul   = (1 + r3).cumprod()
        max_dd  = ((cumul - cumul.cummax()) / cumul.cummax()).min()
        mom6    = (df[ticker].iloc[-1] / df[ticker].iloc[-126] - 1) \
                  if len(df[ticker]) >= 126 else 0

        base_data.append({
            'ticker'          : ticker,
            'zone'            : detect_zone(ticker),
            'sector'          : SECTOR_MAP.get(ticker, 'Other'),
            'annual_return'   : ar3,
            'annual_return_1y': ar1,
            'annual_return_6m': ar6,
            'annual_vol'      : av3,
            'sharpe_combined' : sh_combined,
            'sharpe_3y'       : sh3,
            'sharpe_1y'       : sh1,
            'sharpe_6m'       : sh6,
            'beta'            : beta,
            'alpha'           : alpha,
            'z_score'         : z_final,
            'max_drawdown'    : max_dd,
            'momentum_6m'     : mom6,
        })
    except Exception as e:
        print(f"  Error {ticker}: {e}")

base_df = pd.DataFrame(base_data).set_index('ticker')
print(f"Metrics computed for {len(base_df)} assets")

# ============================================================
# OLS REGRESSIONS
# ============================================================
print("\n==== OLS Regressions ====")

reg_df = base_df.join(fund_df, how='left', rsuffix='_f')
reg_df = reg_df.dropna(subset=[
    'sharpe_combined','alpha','beta','z_score',
    'max_drawdown','momentum_6m',
    'pe_ratio','pb_ratio','ev_ebitda','roe','operating_margin'
])
print(f"OLS observations: {len(reg_df)}")

y  = reg_df['sharpe_combined']
X1 = sm.add_constant(reg_df[['annual_vol','beta','max_drawdown']])
X2 = sm.add_constant(reg_df[['z_score']])
X3 = sm.add_constant(reg_df[[
    'z_score','alpha','beta','momentum_6m','max_drawdown',
    'pe_ratio','pb_ratio','ev_ebitda','roe','operating_margin'
]])
m1 = sm.OLS(y, X1).fit()
m2 = sm.OLS(y, X2).fit()
m3 = sm.OLS(y, X3).fit()

for label, mod in [('Risk only', m1),
                   ('Z-score only', m2),
                   ('Full model', m3)]:
    print(f"\n── Model: {label} ──")
    print(mod.summary())

print("\n==== OLS Comparison ====")
print(pd.DataFrame({
    'Model'   : ['M1 — Risk','M2 — Z-score','M3 — Full'],
    'R²'      : [round(m1.rsquared,4),
                 round(m2.rsquared,4),
                 round(m3.rsquared,4)],
    'Adj. R²' : [round(m1.rsquared_adj,4),
                 round(m2.rsquared_adj,4),
                 round(m3.rsquared_adj,4)],
    'F-stat'  : [round(m1.fvalue,2),
                 round(m2.fvalue,2),
                 round(m3.fvalue,2)],
    'p(F)'    : [round(m1.f_pvalue,6),
                 round(m2.f_pvalue,6),
                 round(m3.f_pvalue,6)],
    'N'       : [int(m1.nobs),int(m2.nobs),int(m3.nobs)]
}).to_string(index=False))

# ============================================================
# SUMMARY STATISTICS — SECTION 3.2 OF THE THESIS
# ============================================================
print("\n" + "="*70)
print("         SUMMARY STATISTICS — SECTION 3.2 OF THE THESIS")
print("="*70)

vars_risk          = ['sharpe_combined','alpha','beta',
                      'z_score','momentum_6m','max_drawdown']
vars_fundamentals  = ['pe_ratio','pb_ratio',
                      'ev_ebitda','roe','operating_margin']

# Full universe
n_total = len(base_df)
print(f"\n── Risk/performance variables ({n_total} assets) ──")
print(base_df[vars_risk].describe().round(4).to_string())

# Fundamentals across the full universe (with NaN)
base_with_fund = base_df.join(fund_df, how='left', rsuffix='_f')
n_fund = base_with_fund[vars_fundamentals].notna().min(axis=1).sum()
print(f"\n── Fundamental variables ({n_total} assets, "
      f"NaN possible) ──")
print(base_with_fund[vars_fundamentals].describe().round(4).to_string())

# OLS dataset (complete observations only)
n_ols = len(reg_df)
print(f"\n── OLS dataset — all variables ({n_ols} complete "
      f"observations) ──")
ols_vars = vars_risk + vars_fundamentals
print(reg_df[ols_vars].describe().round(4).to_string())

# Correlation matrix for key variables
print(f"\n── Correlation matrix — Key variables "
      f"({n_ols} obs.) ──")
corr_vars = ['sharpe_combined','z_score','alpha',
             'momentum_6m','pe_ratio','pb_ratio','roe']
corr_vars_ok = [v for v in corr_vars if v in reg_df.columns]
print(reg_df[corr_vars_ok].corr().round(4).to_string())

# Statistics by geographic zone
print(f"\n── Combined Sharpe by geographic zone "
      f"({n_total} assets) ──")
geo_stats = base_df.groupby('zone')['sharpe_combined'].agg(
    ['count','mean','std','min',
     lambda x: x.quantile(0.25),
     'median',
     lambda x: x.quantile(0.75),
     'max']
).round(4)
geo_stats.columns = ['N','Mean','Std Dev','Min',
                     'Q1','Median','Q3','Max']
print(geo_stats.to_string())

# Statistics by sector
print(f"\n── Combined Sharpe by sector ({n_total} assets) ──")
sec_stats = base_df.groupby('sector')['sharpe_combined'].agg(
    ['count','mean','std','min',
     lambda x: x.quantile(0.25),
     'median',
     lambda x: x.quantile(0.75),
     'max']
).round(4)
sec_stats.columns = ['N','Mean','Std Dev','Min',
                     'Q1','Median','Q3','Max']
sec_stats = sec_stats.sort_values('Mean', ascending=False)
print(sec_stats.to_string())

# Z-score distribution by decile (useful to validate construction)
print(f"\n── Composite Z-score distribution by decile ──")
base_df['z_decile'] = pd.qcut(base_df['z_score'], q=10,
                               labels=False, duplicates='drop')
z_decile_stats = base_df.groupby('z_decile')[
    ['z_score','sharpe_combined','alpha']
].mean().round(4)
z_decile_stats.index = [f"D{i+1}" for i in range(len(z_decile_stats))]
z_decile_stats.columns = ['Mean Z-Score','Mean Sharpe','Mean Alpha']
print(z_decile_stats.to_string())
base_df.drop(columns=['z_decile'], inplace=True)

# Fundamental data availability rate by zone
print(f"\n── Fundamental data availability by zone ──")
avail = base_with_fund.groupby('zone')[vars_fundamentals].apply(
    lambda x: (x.notna().sum() / len(x) * 100).round(1)
)
print(avail.to_string())

print("\n" + "="*70)

# ============================================================
# FUNCTIONS SHARED ACROSS ALL 3 PORTFOLIOS
# ============================================================

def apply_filters_perf(df):
    return df[
        (df['alpha']            >  0.00) &
        (df['beta']             >  0.20) &
        (df['beta']             <  2.50) &
        (df['sharpe_combined']  >  0.30) &
        (df['sharpe_1y']        >  0.00) &
        (df['sharpe_6m']        > -0.30) &
        (df['z_score']          > -2.00) &
        (df['z_score']          <  2.50) &
        (df['max_drawdown']     > -0.60) &
        (df['annual_return']    >  0.00) &
        (df['annual_return_1y'] >  0.00)
    ].copy()


def build_perf_score(df):
    df = df.copy()
    df['sc_alpha']   = normalize(df['alpha'])
    df['sc_sharpe']  = normalize(df['sharpe_combined'])
    df['sc_sh1']     = normalize(df['sharpe_1y'])
    df['sc_sh6']     = normalize(df['sharpe_6m'])
    df['sc_z']       = normalize(df['z_score'])
    df['sc_dd']      = normalize(-df['max_drawdown'])
    df['sc_beta']    = normalize(1 - abs(df['beta'] - 1))
    df['sc_mom']     = normalize(df['momentum_6m'])
    df['score_perf'] = (
        0.20 * df['sc_alpha']  +
        0.20 * df['sc_sharpe'] +
        0.15 * df['sc_sh1']    +
        0.20 * df['sc_sh6']    +
        0.10 * df['sc_z']      +
        0.05 * df['sc_dd']     +
        0.05 * df['sc_beta']   +
        0.05 * df['sc_mom']
    )
    return df


def build_fundamental_score(df_base, df_fund):
    """
    Pure fundamental score based on 5 ratios.
    Academic filter: ROE >= 0 and operating margin >= 0.
    References: Graham (1949), Fama-French (1992).
    """
    df = df_base.join(df_fund, how='left', rsuffix='_f').copy()

    mask = (
        df['roe'].notna()              & (df['roe']             >= 0) &
        df['operating_margin'].notna() & (df['operating_margin'] >= 0) &
        df['pe_ratio'].notna()         &
        df['pb_ratio'].notna()         &
        df['ev_ebitda'].notna()
    )
    df_r    = df[mask].copy()
    n_excl  = len(df) - len(df_r)
    print(f"  Fundamental filter: {len(df_r)} assets retained "
          f"({n_excl} excluded — ROE<0, margin<0, or missing data)")

    def zsc(col):
        s = df_r[col].dropna()
        return (df_r[col] - s.mean()) / s.std()

    df_r['z_pe']     = -zsc('pe_ratio')
    df_r['z_pb']     = -zsc('pb_ratio')
    df_r['z_ev']     = -zsc('ev_ebitda')
    df_r['z_roe']    =  zsc('roe')
    df_r['z_margin'] =  zsc('operating_margin')

    df_r['score_fund'] = df_r[
        ['z_pe','z_pb','z_ev','z_roe','z_margin']
    ].mean(axis=1)

    return df_r['score_fund']


def select_25(candidates, score_col, log_ret_all, n_target=25):
    cands      = candidates.copy()
    selected   = []
    sector_cnt = {s: 0 for s in SECTOR_QUOTAS}

    # Pass 1 — best per sector
    for sector in SECTOR_QUOTAS:
        pool = cands[cands['sector'] == sector]
        if len(pool) > 0:
            best = pool[score_col].idxmax()
            selected.append(best)
            sector_cnt[sector] += 1

    avail    = [t for t in cands.index if t in log_ret_all.columns]
    corr_mat = log_ret_all[avail].corr()

    # Pass 2 — fill with correlation penalty
    remaining = cands[~cands.index.isin(selected)].copy()
    while len(selected) < n_target and len(remaining) > 0:
        dc = {}
        for t in remaining.index:
            if t in corr_mat.index:
                corrs = [abs(corr_mat.loc[s, t])
                         for s in selected
                         if s in corr_mat.index and t in corr_mat.columns]
                dc[t] = 1 - (np.mean(corrs) if corrs else 0.5)
            else:
                dc[t] = 0.5

        dc_norm                  = normalize(pd.Series(dc))
        remaining['sc_decorr']   = dc_norm
        remaining['sc_adjusted'] = (0.70 * remaining[score_col] +
                                    0.30 * remaining['sc_decorr'])

        best = None
        for t in remaining.sort_values('sc_adjusted', ascending=False).index:
            sec = remaining.loc[t, 'sector']
            if sector_cnt.get(sec, 0) < SECTOR_QUOTAS.get(sec, 2):
                best = t
                break
        if best is None:
            best = remaining['sc_adjusted'].idxmax()

        sec_b = remaining.loc[best, 'sector']
        sector_cnt[sec_b] = sector_cnt.get(sec_b, 0) + 1
        selected.append(best)
        remaining = remaining.drop(best)

    # Pass 3 — unconstrained fill
    if len(selected) < n_target:
        rest = cands[~cands.index.isin(selected)]
        for t in rest.nlargest(n_target - len(selected), score_col).index:
            selected.append(t)

    return selected[:n_target]


def optimize_portfolio(symbols_f, mean_ret, cov_mat, rf, lam):
    n       = len(symbols_f)
    cov_np  = cov_mat.loc[symbols_f, symbols_f].values
    mean_np = mean_ret[symbols_f].values

    def neg_sharpe(w):
        ret = float(w @ mean_np)
        vol = float(np.sqrt(w @ cov_np @ w))
        return -((ret - rf) / vol - lam * float(np.sum(w**2))) \
               if vol > 0 else 0.0

    def port_var_reg(w):
        return float(w @ (cov_np + np.eye(n)*1e-8) @ w)

    constraints = [
        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.sum(w >= 0.01) - 10}
    ]
    bounds = tuple((0.001, 0.30) for _ in symbols_f)

    np.random.seed(42)
    mc_w  = np.random.random((50_000, n))
    mc_w /= mc_w.sum(axis=1, keepdims=True)
    mc_r  = mc_w @ mean_np
    mc_v  = np.sqrt(np.einsum('ij,jk,ik->i', mc_w, cov_np, mc_w))
    mc_s  = (mc_r - rf) / mc_v
    mc_df = pd.DataFrame({
        'Return': mc_r, 'Volatility': mc_v, 'Sharpe': mc_s
    })

    np.random.seed(123)
    starts = [np.ones(n) / n]
    for idx in np.argsort(mc_s)[-24:]:
        starts.append(mc_w[idx])
    for _ in range(25):
        w = np.random.dirichlet(np.ones(n))
        starts.append(w / w.sum())

    best_pen, w_opt = -np.inf, None
    for w0 in starts:
        try:
            res = minimize(neg_sharpe, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-15})
            if res.success and -res.fun > best_pen:
                best_pen, w_opt = -res.fun, res.x
        except:
            continue

    best_var, w_mv = np.inf, None
    for w0 in starts[:25]:
        try:
            res = minimize(port_var_reg, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-15})
            if res.success and res.fun < best_var:
                best_var, w_mv = res.fun, res.x
        except:
            continue

    return (pd.Series(w_opt, index=symbols_f),
            pd.Series(w_mv if w_mv is not None else w_opt, index=symbols_f),
            mc_df)


def portfolio_metrics(w, daily_f, cov_f, market_raw, rf,
                      investment=1_000_000):
    port_ret = daily_f @ w
    ann_ret  = port_ret.mean() * 252
    ann_vol  = port_ret.std()  * np.sqrt(252)
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    alpha_p = 0.0
    if market_raw is not None:
        common = port_ret.index.intersection(market_raw.index)
        if len(common) > 60:
            p = port_ret.loc[common]
            m = market_raw.loc[common]
            b = p.cov(m) / m.var()
            alpha_p = ann_ret - rf - b * (m.mean() * 252 - rf)

    cumul   = (1 + port_ret).cumprod()
    max_dd  = ((cumul - cumul.cummax()) / cumul.cummax()).min()
    var_1d  = -investment * port_ret.quantile(0.01)
    perf_6m = float((1 + port_ret.tail(126)).prod() - 1)
    perf_3y = float(cumul.iloc[-1] - 1)
    n       = len(w)
    avg_c   = (daily_f.corr().values.sum() - n) / (n*(n-1)) if n > 1 else 0

    return {
        'Annualized Return'    : ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio'         : sharpe,
        'Jensen Alpha'         : alpha_p,
        'Maximum Drawdown'     : max_dd,
        '1d VaR 99%'          : var_1d,
        '6-Month Perf.'        : perf_6m,
        '3-Year Perf.'         : perf_3y,
        'Avg. Asset Corr.'     : avg_c,
    }

# ============================================================
# ══════════════════════════════════════════════════════════
# PORTFOLIO 1 — PURE FUNDAMENTAL APPROACH
# ══════════════════════════════════════════════════════════
# ============================================================
print("\n" + "="*65)
print("   PORTFOLIO 1 — PURE FUNDAMENTAL APPROACH")
print("="*65)

print("\nBuilding fundamental score...")
score_fund_all        = build_fundamental_score(base_df, fund_df)
base_df['score_fund'] = score_fund_all

p1_candidates = base_df[base_df['score_fund'].notna()].copy()
p1_candidates['sector'] = p1_candidates['sector'].fillna('Other')
print(f"P1 candidates (profitable, data available): {len(p1_candidates)}")

p1_tickers = select_25(p1_candidates, 'score_fund', log_returns)
p1_zones   = [detect_zone(t) for t in p1_tickers]
p1_sectors = [SECTOR_MAP.get(t,'Other') for t in p1_tickers]

print(f"\n✅ P1 — 25 selected assets (Fundamental):")
print(f"{'Ticker':<12} {'Zone':<8} {'Sector':<15} "
      f"{'FundScore':>10} {'P/E':>7} {'P/B':>6} "
      f"{'EV/EBITDA':>10} {'ROE':>7} {'Margin':>7}")
print("-" * 75)
for t in p1_tickers:
    sf   = p1_candidates.loc[t,'score_fund'] \
           if t in p1_candidates.index else np.nan
    pe   = fund_df.loc[t,'pe_ratio']         if t in fund_df.index else np.nan
    pb   = fund_df.loc[t,'pb_ratio']         if t in fund_df.index else np.nan
    ev   = fund_df.loc[t,'ev_ebitda']        if t in fund_df.index else np.nan
    roe  = fund_df.loc[t,'roe']              if t in fund_df.index else np.nan
    marg = fund_df.loc[t,'operating_margin'] if t in fund_df.index else np.nan
    print(f"{t:<12} {detect_zone(t):<8} "
          f"{SECTOR_MAP.get(t,'Other'):<15} "
          f"{sf:>10.4f} {pe:>7.1f} {pb:>6.2f} "
          f"{ev:>10.1f} {roe:>7.3f} {marg:>7.3f}")

p1_daily   = daily_returns[[t for t in p1_tickers
                             if t in daily_returns.columns]]
p1_tickers = list(p1_daily.columns)
w_p1, w_p1_mv, mc_p1 = optimize_portfolio(
    p1_tickers, mean_ewma, cov_lw_matrix, rf, LAMBDA_CONCENTRATION
)

r_p1  = float(w_p1 @ mean_ewma[p1_tickers])
v_p1  = float(np.sqrt(
    w_p1 @ cov_lw_matrix.loc[p1_tickers,p1_tickers] @ w_p1))
sh_p1 = (r_p1 - rf) / v_p1

print(f"\nMax-Sharpe Weights P1:")
p1_w_df = pd.DataFrame({
    'Ticker': p1_tickers,
    'Zone'  : p1_zones[:len(p1_tickers)],
    'Sector': p1_sectors[:len(p1_tickers)],
    'Weight': w_p1.values
}).sort_values('Weight', ascending=False)
p1_w_df['Weight%'] = p1_w_df['Weight'].map(lambda x: f"{x*100:.4f}%")
print(p1_w_df[['Ticker','Zone','Sector','Weight%']].to_string(index=False))
print(f"EWMA Sharpe P1: {sh_p1:.4f}")

met_p1 = portfolio_metrics(
    w_p1, p1_daily,
    cov_lw_matrix.loc[p1_tickers,p1_tickers],
    market_raw, rf
)

# ============================================================
# ══════════════════════════════════════════════════════════
# PORTFOLIO 2 — RISK-ADJUSTED PERFORMANCE
# ══════════════════════════════════════════════════════════
# ============================================================
print("\n" + "="*65)
print("   PORTFOLIO 2 — RISK-ADJUSTED PERFORMANCE")
print("="*65)

p2_filtered = apply_filters_perf(base_df)
p2_filtered = build_perf_score(p2_filtered)
print(f"Eligible assets P2: {len(p2_filtered)}")

p2_tickers = select_25(p2_filtered, 'score_perf', log_returns)
p2_zones   = [detect_zone(t) for t in p2_tickers]
p2_sectors = [SECTOR_MAP.get(t,'Other') for t in p2_tickers]

print(f"\n✅ P2 — 25 selected assets (Performance):")
print(f"{'Ticker':<12} {'Zone':<8} {'Sector':<15} "
      f"{'Score':>8} {'Sharpe':>8} {'Alpha':>8} {'Z':>6}")
print("-" * 65)
for t in p2_tickers:
    if t in p2_filtered.index:
        row = p2_filtered.loc[t]
        print(f"{t:<12} {detect_zone(t):<8} "
              f"{SECTOR_MAP.get(t,'Other'):<15} "
              f"{row['score_perf']:>8.4f} "
              f"{row['sharpe_combined']:>8.4f} "
              f"{row['alpha']:>8.4f} "
              f"{row['z_score']:>6.3f}")

p2_daily   = daily_returns[[t for t in p2_tickers
                             if t in daily_returns.columns]]
p2_tickers = list(p2_daily.columns)
w_p2, w_p2_mv, mc_p2 = optimize_portfolio(
    p2_tickers, mean_ewma, cov_lw_matrix, rf, LAMBDA_CONCENTRATION
)

r_p2  = float(w_p2 @ mean_ewma[p2_tickers])
v_p2  = float(np.sqrt(
    w_p2 @ cov_lw_matrix.loc[p2_tickers,p2_tickers] @ w_p2))
sh_p2 = (r_p2 - rf) / v_p2

print(f"\nMax-Sharpe Weights P2:")
p2_w_df = pd.DataFrame({
    'Ticker': p2_tickers,
    'Zone'  : p2_zones[:len(p2_tickers)],
    'Sector': p2_sectors[:len(p2_tickers)],
    'Weight': w_p2.values
}).sort_values('Weight', ascending=False)
p2_w_df['Weight%'] = p2_w_df['Weight'].map(lambda x: f"{x*100:.4f}%")
print(p2_w_df[['Ticker','Zone','Sector','Weight%']].to_string(index=False))
print(f"EWMA Sharpe P2: {sh_p2:.4f}")

met_p2 = portfolio_metrics(
    w_p2, p2_daily,
    cov_lw_matrix.loc[p2_tickers,p2_tickers],
    market_raw, rf
)

# ============================================================
# ══════════════════════════════════════════════════════════
# PORTFOLIO 3 — COMBINED APPROACH (50% Fund + 50% Perf)
# ══════════════════════════════════════════════════════════
# ============================================================
print("\n" + "="*65)
print("   PORTFOLIO 3 — COMBINED APPROACH")
print("="*65)

p3_filtered = apply_filters_perf(base_df)
p3_filtered = build_perf_score(p3_filtered)
p3_filtered['score_fund'] = score_fund_all.reindex(p3_filtered.index)

# Assets without a valid fundamental score → imputed with median
median_fund = p3_filtered['score_fund'].median()
p3_filtered['score_fund_filled'] = p3_filtered['score_fund'].fillna(
    median_fund
)
p3_filtered['score_fund_norm'] = normalize(p3_filtered['score_fund_filled'])
p3_filtered['score_perf_norm'] = normalize(p3_filtered['score_perf'])
p3_filtered['score_combined']  = (
    0.50 * p3_filtered['score_fund_norm'] +
    0.50 * p3_filtered['score_perf_norm']
)

n_p3_fund = p3_filtered['score_fund'].notna().sum()
print(f"Eligible assets P3: {len(p3_filtered)}")
print(f"  of which {n_p3_fund} with a positive fundamental score "
      f"({len(p3_filtered)-n_p3_fund} with imputed median)")

p3_tickers = select_25(p3_filtered, 'score_combined', log_returns)
p3_zones   = [detect_zone(t) for t in p3_tickers]
p3_sectors = [SECTOR_MAP.get(t,'Other') for t in p3_tickers]

print(f"\n✅ P3 — 25 selected assets (Combined):")
print(f"{'Ticker':<12} {'Zone':<8} {'Sector':<15} "
      f"{'CombScore':>10} {'FundScore':>10} {'PerfScore':>10}")
print("-" * 65)
for t in p3_tickers:
    if t in p3_filtered.index:
        row = p3_filtered.loc[t]
        print(f"{t:<12} {detect_zone(t):<8} "
              f"{SECTOR_MAP.get(t,'Other'):<15} "
              f"{row['score_combined']:>10.4f} "
              f"{row['score_fund_norm']:>10.4f} "
              f"{row['score_perf_norm']:>10.4f}")

p3_daily   = daily_returns[[t for t in p3_tickers
                             if t in daily_returns.columns]]
p3_tickers = list(p3_daily.columns)
w_p3, w_p3_mv, mc_p3 = optimize_portfolio(
    p3_tickers, mean_ewma, cov_lw_matrix, rf, LAMBDA_CONCENTRATION
)

r_p3  = float(w_p3 @ mean_ewma[p3_tickers])
v_p3  = float(np.sqrt(
    w_p3 @ cov_lw_matrix.loc[p3_tickers,p3_tickers] @ w_p3))
sh_p3 = (r_p3 - rf) / v_p3

print(f"\nMax-Sharpe Weights P3:")
p3_w_df = pd.DataFrame({
    'Ticker': p3_tickers,
    'Zone'  : p3_zones[:len(p3_tickers)],
    'Sector': p3_sectors[:len(p3_tickers)],
    'Weight': w_p3.values
}).sort_values('Weight', ascending=False)
p3_w_df['Weight%'] = p3_w_df['Weight'].map(lambda x: f"{x*100:.4f}%")
print(p3_w_df[['Ticker','Zone','Sector','Weight%']].to_string(index=False))
print(f"EWMA Sharpe P3: {sh_p3:.4f}")

met_p3 = portfolio_metrics(
    w_p3, p3_daily,
    cov_lw_matrix.loc[p3_tickers,p3_tickers],
    market_raw, rf
)

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
fmt = {
    'Annualized Return'    : '{:.4%}',
    'Annualized Volatility': '{:.4%}',
    'Sharpe Ratio'         : '{:.4f}',
    'Jensen Alpha'         : '{:.4%}',
    'Maximum Drawdown'     : '{:.4%}',
    '1d VaR 99%'          : '${:,.2f}',
    '6-Month Perf.'        : '{:.4%}',
    '3-Year Perf.'         : '{:.4%}',
    'Avg. Asset Corr.'     : '{:.4f}',
}

print("\n" + "="*75)
print("              COMPARISON TABLE — 3 PORTFOLIOS")
print("="*75)
print(f"\n{'Metric':<25} {'P1 — Fundamental':>20} "
      f"{'P2 — Risk-Adj. Perf.':>20} {'P3 — Combined':>20}")
print("-" * 87)
for met in fmt:
    f = fmt[met]
    print(f"{met:<25} "
          f"{f.format(met_p1[met]):>20} "
          f"{f.format(met_p2[met]):>20} "
          f"{f.format(met_p3[met]):>20}")

sharpes = {
    'P1 — Fundamental'    : met_p1['Sharpe Ratio'],
    'P2 — Risk-Adj. Perf.': met_p2['Sharpe Ratio'],
    'P3 — Combined'       : met_p3['Sharpe Ratio'],
}
best = max(sharpes, key=sharpes.get)
print(f"\n  🏆 Best historical Sharpe: {best} "
      f"({sharpes[best]:.4f})")
print(f"\n  EWMA Sharpe: "
      f"P1={sh_p1:.4f} | P2={sh_p2:.4f} | P3={sh_p3:.4f}")
print(f"  ⚠️  EWMA Sharpe may overestimate — "
      f"historical Sharpe is the reference.")

# ============================================================
# CHARTS
# ============================================================
colors = {'P1':'#4FC3F7', 'P2':'#81C784', 'P3':'#FFB74D'}

# 1 — Cumulative performance comparison
fig     = go.Figure(layout=par84.layout)
idx_ref = None
for label, w, daily, col in [
    ('P1 — Fundamental',          w_p1, p1_daily, colors['P1']),
    ('P2 — Risk-Adj. Performance', w_p2, p2_daily, colors['P2']),
    ('P3 — Combined',              w_p3, p3_daily, colors['P3']),
]:
    port_r = daily @ w
    cumul  = (1 + port_r).cumprod()
    norm   = (cumul - 1) * 100
    fig.add_scatter(x=norm.index, y=norm.values, mode='lines',
                    name=label, line=dict(color=col, width=2.5))
    if idx_ref is None:
        idx_ref = cumul.index

if idx_ref is not None:
    for bname, braw, bcol, bdash in [
        ('S&P 500',  sp500_raw,  'orange', 'dash'),
        ('STOXX 50', stoxx_raw,  'cyan',   'dot'),
        ('Nikkei',   nikkei_raw, 'red',    'dashdot'),
    ]:
        if braw is not None:
            c  = braw.reindex(idx_ref).ffill()
            nb = (c / c.iloc[0] - 1) * 100
            fig.add_scatter(x=nb.index, y=nb.values, mode='lines',
                            name=bname,
                            line=dict(color=bcol, width=1.5, dash=bdash))

fig.update_layout(
    title='Cumulative Performance — 3 Portfolios vs Benchmarks',
    xaxis_title='Date', yaxis_title='Performance (%)'
)
fig.write_html(images+"1.html", config=par84.config, include_plotlyjs="cdn")

# 2 — Synthetic comparison table
metrics_labels = list(fmt.keys())
fmt_vals = [
    [fmt[m].format(met_p1[m]),
     fmt[m].format(met_p2[m]),
     fmt[m].format(met_p3[m])]
    for m in metrics_labels
]

fig = go.Figure(layout=par84.layout)
fig.add_table(
    header=dict(
        values=['<b>Metric</b>',
                '<b>P1 — Fundamental</b>',
                '<b>P2 — Risk-Adj. Perf.</b>',
                '<b>P3 — Combined</b>'],
        fill_color='#1e3a5f',
        font=dict(color='white', size=12),
        align='left'
    ),
    cells=dict(
        values=[
            metrics_labels,
            [v[0] for v in fmt_vals],
            [v[1] for v in fmt_vals],
            [v[2] for v in fmt_vals],
        ],
        fill_color='#0d2137',
        font=dict(color='white', size=11),
        align=['left','right','right','right']
    )
)
fig.update_layout(title='Comparison Table — 3 Portfolios')
fig.write_html(images+"2.html", config=par84.config, include_plotlyjs="cdn")

# 3 — Efficient frontier
fig = go.Figure(layout=par84.layout)

for df_mc, col_mc, name_mc in [
    (mc_p1, colors['P1'], 'MC P1'),
    (mc_p2, colors['P2'], 'MC P2'),
    (mc_p3, colors['P3'], 'MC P3'),
]:
    fig.add_scatter(
        x=df_mc['Volatility'], y=df_mc['Return'],
        mode='markers',
        marker=dict(color=col_mc, size=1.5, opacity=0.15),
        name=name_mc
    )

for label, w, tickers, col in [
    ('P1 — Fundamental',   w_p1, p1_tickers, colors['P1']),
    ('P2 — Risk-Adj. Perf.',w_p2, p2_tickers, colors['P2']),
    ('P3 — Combined',       w_p3, p3_tickers, colors['P3']),
]:
    cov_f  = cov_lw_matrix.loc[tickers, tickers]
    mean_f = mean_ewma[tickers]
    r = float(w @ mean_f)
    v = float(np.sqrt(w @ cov_f @ w))
    s = (r - rf) / v
    fig.add_scatter(
        x=[v], y=[r], mode='markers',
        marker=dict(color=col, size=18, symbol='star',
                    line=dict(color='white', width=1)),
        name=f'{label} (Sharpe={s:.2f})'
    )

fig.update_layout(
    title='Efficient Frontier — 3 Portfolios',
    xaxis_title='Volatility (EWMA)',
    yaxis_title='Return (EWMA)'
)
fig.write_html(images+"3.html", config=par84.config, include_plotlyjs="cdn")

# 4-6 — Individual allocations
for i, (label, w_arr, tickers, sectors) in enumerate([
    ('P1 — Fundamental',          w_p1, p1_tickers, p1_sectors),
    ('P2 — Risk-Adj. Performance', w_p2, p2_tickers, p2_sectors),
    ('P3 — Combined',              w_p3, p3_tickers, p3_sectors),
], start=4):
    labels_pie = [
        f"{tickers[j]}<br>{sectors[j]}"
        for j in range(len(tickers))
    ]
    fig = go.Figure(layout=par84.layout)
    fig.add_pie(labels=labels_pie, values=w_arr.values,
                hole=0.5, textposition='inside',
                textinfo='label+percent')
    fig.update_layout(title=f'Allocation — {label}')
    fig.write_html(images+f"{i}.html",
                   config=par84.config, include_plotlyjs="cdn")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*75)
print("                   THESIS SUMMARY")
print("="*75)

print(f"\n  📌 COMMON METHODOLOGY")
print(f"  Universe                        : {len(symbols)} companies")
print(f"  Assets retained (95% data)      : {len(symbols_clean)}")
print(f"  Period                          : 3 years")
print(f"  Covariance                      : Ledoit-Wolf ({lw.shrinkage_:.4f})")
print(f"  Expected returns                : EWMA (half-life 252d)")
print(f"  Optimization                    : Max-Sharpe + Herfindahl "
      f"(λ={LAMBDA_CONCENTRATION})")
print(f"  Selection                       : Sector quotas + "
      f"correlation penalty")
print(f"  Weight constraints              : [0.1%; 30%]")

n_fund_valide = base_df['score_fund'].notna().sum()
print(f"\n  📐 FUNDAMENTAL FILTER P1")
print(f"  Assets with ROE≥0 and margin≥0  : {n_fund_valide}")
print(f"  Excluded (ROE<0 or margin<0)    : "
      f"{len(symbols_clean) - n_fund_valide}")
print(f"  Justification                   : Graham (1949), "
      f"Fama-French (1992)")

print(f"\n  📊 OLS REGRESSIONS ({len(reg_df)} observations)")
print(f"  M1 — Risk only     : R²={m1.rsquared:.4f}, "
      f"F={m1.fvalue:.2f}, p={m1.f_pvalue:.4f}")
print(f"  M2 — Z-score only  : R²={m2.rsquared:.4f}, "
      f"coef={m2.params['z_score']:.4f}, "
      f"p={m2.pvalues['z_score']:.4f}")
print(f"  M3 — Full model    : R²={m3.rsquared:.4f}, "
      f"F={m3.fvalue:.2f}, p={m3.f_pvalue:.4f}")

print(f"\n  📈 COMPARISON OF 3 PORTFOLIOS")
print(f"  {'Metric':<26} {'P1 Fund':>12} "
      f"{'P2 Perf':>12} {'P3 Comb':>12}")
print(f"  {'-'*62}")
rows = [
    ('Historical Sharpe',
     f"{met_p1['Sharpe Ratio']:.4f}",
     f"{met_p2['Sharpe Ratio']:.4f}",
     f"{met_p3['Sharpe Ratio']:.4f}"),
    ('Annualized Return',
     f"{met_p1['Annualized Return']:.4%}",
     f"{met_p2['Annualized Return']:.4%}",
     f"{met_p3['Annualized Return']:.4%}"),
    ('Annualized Volatility',
     f"{met_p1['Annualized Volatility']:.4%}",
     f"{met_p2['Annualized Volatility']:.4%}",
     f"{met_p3['Annualized Volatility']:.4%}"),
    ('Jensen Alpha',
     f"{met_p1['Jensen Alpha']:.4%}",
     f"{met_p2['Jensen Alpha']:.4%}",
     f"{met_p3['Jensen Alpha']:.4%}"),
    ('Maximum Drawdown',
     f"{met_p1['Maximum Drawdown']:.4%}",
     f"{met_p2['Maximum Drawdown']:.4%}",
     f"{met_p3['Maximum Drawdown']:.4%}"),
    ('1d VaR 99%',
     f"${met_p1['1d VaR 99%']:,.0f}",
     f"${met_p2['1d VaR 99%']:,.0f}",
     f"${met_p3['1d VaR 99%']:,.0f}"),
    ('3-Year Cumulative Perf.',
     f"{met_p1['3-Year Perf.']:.4%}",
     f"{met_p2['3-Year Perf.']:.4%}",
     f"{met_p3['3-Year Perf.']:.4%}"),
    ('Avg. Asset Correlation',
     f"{met_p1['Avg. Asset Corr.']:.4f}",
     f"{met_p2['Avg. Asset Corr.']:.4f}",
     f"{met_p3['Avg. Asset Corr.']:.4f}"),
]
for r in rows:
    print(f"  {r[0]:<26} {r[1]:>12} {r[2]:>12} {r[3]:>12}")

print(f"\n  🏆 Best historical Sharpe: {best} ({sharpes[best]:.4f})")
print(f"\n  EWMA Sharpe: "
      f"P1={sh_p1:.4f} | P2={sh_p2:.4f} | P3={sh_p3:.4f}")
print(f"  ⚠️  EWMA Sharpe may overestimate — "
      f"historical Sharpe is the reference.")
print("="*75)