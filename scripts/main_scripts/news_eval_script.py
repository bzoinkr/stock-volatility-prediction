import os

os.system("python scripts/train_news/032_fetch_market_volatility_compound_news.py")
os.system("python scripts/train_news/042_fetch_news_train.py")
os.system("python scripts/12_execute_finbert.py")
os.system("python scripts/22_clean_news_data.py")
os.system("python scripts/42_createModel_news.py")
os.system("python scripts/43_evaluate_news.py")