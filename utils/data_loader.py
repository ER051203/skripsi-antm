# utils/data_loader.py
"""
Fungsi untuk mengambil data saham dari Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_stock_data(ticker="ANTM.JK", start_date="2015-01-01", end_date=None):
    """
    Download data saham dari Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Kode saham (default: ANTM.JK)
    start_date : str
        Tanggal mulai (format: YYYY-MM-DD)
    end_date : str or None
        Tanggal akhir (None = sampai hari ini)
    
    Returns:
    --------
    pandas.DataFrame with Date and Close columns
    """
    try:
        logger.info(f"Mendownload data {ticker}...")
        
        # Set end date to today if not specified
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"Tidak ada data untuk {ticker}")
        
        # Reset index to get Date as column
        data.reset_index(inplace=True)
        
        # Keep only Date and Close
        data = data[['Date', 'Close']]
        
        # Handle missing values
        data['Close'].fillna(method='ffill', inplace=True)
        data['Close'].fillna(method='bfill', inplace=True)
        
        logger.info(f"Berhasil download {len(data)} data dari {data['Date'].min().date()} sampai {data['Date'].max().date()}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

def get_stock_info(ticker="ANTM.JK"):
    """
    Mendapatkan informasi perusahaan
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', 'PT Aneka Tambang Tbk'),
            'sector': info.get('sector', 'Mining'),
            'industry': info.get('industry', 'Metal Mining'),
            'website': info.get('website', '#'),
            'currency': info.get('currency', 'IDR')
        }
    except:
        return {
            'name': 'PT Aneka Tambang Tbk',
            'sector': 'Mining',
            'industry': 'Metal Mining',
            'website': '#',
            'currency': 'IDR'
        }