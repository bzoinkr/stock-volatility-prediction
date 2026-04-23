import os

os.system("python scripts/03_fetch_market_volatility_target.py")
os.system("python scripts/02_fetch_news.py")
os.system("python scripts/12_execute_finbert.py")
os.system("python scripts/22_clean_news_data.py")
os.system("python scripts/52_predict_news.py")