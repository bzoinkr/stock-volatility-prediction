import os

os.system("python scripts/train_social/032_fetch_market_volatility_compound.py")
os.system("python scripts/train_social/042_fetch_social_train.py")
os.system("python scripts/train_social/112_execute_vader_train.py")
os.system("python scripts/train_social/212_clean_social_data_train.py")
os.system("python scripts/40_createModel_social.py")
os.system("python scripts/41_evaluate_social.py")
