import unittest
import logging
import pandas as pd

from loaders.cognicity_loader import CognicityLoader
from nlp.bow_labeler import BowLabeler

from chennai_config import config


class BowLabelerTest(unittest.TestCase):
    """
    Integration tests for bag of words labeler

    In order for these to run, a real cognicity db has to be running on
    port 5432 with username postgres and password postgres
    These are integration tests, not just unit tests. As such,
    those that require a database will not run on travis ci

    Also, test methods need to start with "test_" in order to be run
    """

    def setUp(self):
        self.loader = CognicityLoader(config)
        # make sure there's a logger and a loader
        self.assertIsInstance(self.loader.logger, logging.Logger)
        self.assertTrue(self.loader)

        self.labeler = BowLabeler(config, self.loader)
        self.assertIsInstance(self.labeler.logger, logging.Logger)
        self.assertTrue(self.labeler)

    def test_get_labels_one(self):
        MOCK_TEXT = "sentence test"
        MOCK_PKEY = 125
        mock_df = pd.DataFrame({"text": [MOCK_TEXT]},
                               index=[MOCK_PKEY])
        mock_df.rename_axis("pkey", inplace=True)
        labs = self.labeler.get_labels(mock_df)
        self.assertIsInstance(labs[MOCK_PKEY], list)
        self.assertTrue(labs[MOCK_PKEY][0] == "sentence")

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main()
