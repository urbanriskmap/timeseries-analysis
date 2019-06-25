import unittest
import logging
import shutil
import os
import types

from sqlalchemy import create_engine

from cognicity_image_loader import CognicityImageLoader

IMG_FOLDER_PREFIX = "./test_img_folder_"
TEST_LOCATION = "id"

class CognicityImageLoaderTest(unittest.TestCase):
    """
    In order for these to run, a real cognicity db has to be running on
    port 5432 with username postgres and password postgres
    """

    def setUp(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler('test_log_filename.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.debug('This is a test log message.')

        DATABASE = "cognicity"
        ID_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)
        configObj = { "database_engine": ID_ENGINE, 
                    "database_name": "cognicity", 
                    "location": TEST_LOCATION, 
                    "img_folder_prefix": IMG_FOLDER_PREFIX,
                    "logger": logger}

        self.loader = CognicityImageLoader(configObj)
        self.assertTrue(self.loader)

    def test_get_image_urls(self):
        urls = self.loader.get_image_urls()
        self.assertIsInstance(urls, dict)
        one_pkey = list(urls.keys())[0]
        self.assertIsInstance(urls[one_pkey], str)

        # make sure that the url points to a resource?
        # or test with a real pkey from indonesia?

    def test_fetch_images(self):
        # ... let's remove the test folder if it exists
        shutil.rmtree(IMG_FOLDER_PREFIX + TEST_LOCATION, ignore_errors=True)

        # make sure that the img is downloaded
        # first test img of riskmap favicon
        realLoader = self.loader.get_image_urls
        def mock_get_image_urls(self):
            return { 0: "https://riskmap.in/assets/logos/url_logo.png"}

        # replace the get_image_urls function with our mock one 
        self.loader.get_image_urls = types.MethodType(mock_get_image_urls, CognicityImageLoader)
        self.loader.fetch_images()

        # make sure folder gets created
        self.assertTrue(os.path.exists(IMG_FOLDER_PREFIX + TEST_LOCATION))

        img_path = IMG_FOLDER_PREFIX+TEST_LOCATION + "/0.jpeg"
        # make sure the image gets created
        for _, __, files in os.walk(IMG_FOLDER_PREFIX + TEST_LOCATION):
            self.assertTrue(len(files) > 0)
            for name in files:
                self.assertEqual(name, "0.jpeg")
                self.assertTrue(os.stat(img_path).st_size > 0)

        

        # clean up after ourselves
        shutil.rmtree(IMG_FOLDER_PREFIX + TEST_LOCATION, ignore_errors=True)
        pass







if __name__ == "__main__":
    unittest.main()
