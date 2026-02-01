from _bootstrap import *

from common.config import load_config

cfg = load_config("run.yaml")

print("Project name:", cfg["project"]["name"])
print("Tickers:", cfg["universe"]["tickers"])
