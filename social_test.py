import os


os.system("python scripts/03_fetch_market_volatility_target.py")
os.system("python scripts/04_fetch_social_target.py")
os.system("python scripts/11_execute_vader.py")
os.system("python scripts/21_clean_social_data.py")
#os.system("python scripts/40_createModel_social.py")
os.system("python scripts/51_predict_social.py")