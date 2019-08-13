import unittest
import logging
import os
import pandas as pd
import shutil

from sqlalchemy import create_engine

from loaders.cognicity_loader import CognicityLoader

IMG_FOLDER_PREFIX = "./test_text_folder_"
TEST_LOCATION = "id"

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

TEST_LOG_FILENAME = "test_log_filename.log"
LOG_PATH = os.path.join(IMG_FOLDER_PREFIX, TEST_LOG_FILENAME)

fh = logging.FileHandler(TEST_LOG_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

LOGGER.debug('This is a test log message.')


class CognicityTextLoaderTest(unittest.TestCase):
    """
    Integration tests for bag of words labeler

    In order for these to run, a real cognicity db has to be running on
    port 5432 with username postgres and password postgres
    These are integration tests, not just unit tests. As such,
    those that require a database will not run on travis ci

    Also, test methods need to start with "test_" in order to be run
    """

    def setUp(self):

        DATABASE = "cognicity"
        ID_ENGINE = create_engine(
                    "postgresql://postgres:postgres@localhost:5432/"
                    + DATABASE)
        configObj = {
                "database_engine": ID_ENGINE,
                "database_name": "cognicity",
                "location": TEST_LOCATION,
                "data_folder_prefix": IMG_FOLDER_PREFIX,
                "logger": LOGGER}

        self.loader = CognicityLoader(configObj)
        self.assertTrue(self.loader)

    @unittest.skipIf(os.environ.get("TRAVIS"),
                     "skipping integration test")
    def test_get_texts(self):
        text_df = self.loader.get_texts()
        self.assertIsInstance(text_df, pd.DataFrame)
        one_row = text_df.iloc[0]
        self.assertIsInstance(one_row["text"], str)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(IMG_FOLDER_PREFIX)


if __name__ == "__main__":
    unittest.main()
