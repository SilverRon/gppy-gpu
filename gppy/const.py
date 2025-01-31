import os

os.environ["SCRIPT_DIR"] = "/data/pipeline_reform/gppy-gpu"
# os.environ["RAWDATA_DIR"] = "/lyman/data1/obsdata"
os.environ["RAWDATA_DIR"] = "/data/pipeline_reform/obsdata_test"
os.environ["PROCESSED_DIR"] = "/data/pipeline_reform/processed_test_light"
os.environ["MASTER_FRAME_DIR"] = "/data/pipeline_reform/master_frame_test"
os.environ["FACTORY_DIR"] = "/data/pipeline_reform/factory_test"
os.environ["SLACK_TOKEN"] = "xoxb-463145837843-2139189666645-fn76rVNOhiDRuWOIhfxW4RTD"

SCRIPT_DIR = os.environ["SCRIPT_DIR"]
RAWDATA_DIR = os.environ["RAWDATA_DIR"]
PROCESSED_DIR = os.environ["PROCESSED_DIR"]
MASTER_FRAME_DIR = os.environ["MASTER_FRAME_DIR"]
FACTORY_DIR = os.environ["FACTORY_DIR"]
SLACK_TOKEN = os.environ["SLACK_TOKEN"]
