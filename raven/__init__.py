from os import environ
from os.path import abspath, dirname, join


ROOT_DIR = dirname(abspath(__file__))
CACHE_DIR = abspath(join(ROOT_DIR, "..", "cache"))

environ["HF_DATASETS_CACHE"] = CACHE_DIR
