import unittest
import logging
import shutil
import os
import types

from sqlalchemy import create_engine

from cognicity_image_loader import CognicityImageLoader

IMG_FOLDER_PREFIX = "./test_img_folder_"
TEST_LOCATION = "id"

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('test_log_filename.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

LOGGER.debug('This is a test log message.')


class CognicityImageLoaderTest(unittest.TestCase):
    """
    In order for these to run, a real cognicity db has to be running on
    port 5432 with username postgres and password postgres
    These are integration tests, not just unit tests. As such, 
    those that require a database will not run on travis ci
    """

    def setUp(self):

        DATABASE = "cognicity"
        ID_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)
        configObj = { "database_engine": ID_ENGINE, 
                    "database_name": "cognicity", 
                    "location": TEST_LOCATION, 
                    "img_folder_prefix": IMG_FOLDER_PREFIX,
                    "logger": LOGGER}

        self.loader = CognicityImageLoader(configObj)
        self.assertTrue(self.loader)

    @unittest.skipIf(os.environ.get("TRAVIS"),
            "skipping integration test")
    def test_get_image_urls(self):
        urls = self.loader.get_image_urls()
        self.assertIsInstance(urls, dict)
        one_pkey = list(urls.keys())[0]
        self.assertIsInstance(urls[one_pkey], str)

        # make sure that the string looks like a url
        self.assertTrue("http" in urls[one_pkey])

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

    def test_fetch_images_dif_location(self):
        DATABASE = "cognicity"
        NEW_TEST_LOCATION = "new_test_location"
        ID_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)
        configObj = { "database_engine": ID_ENGINE, 
                    "database_name": "cognicity", 
                    "img_folder_prefix": IMG_FOLDER_PREFIX,
                    "location": NEW_TEST_LOCATION, 
                    "logger": LOGGER}

        mut_loader = CognicityImageLoader(configObj)

        # ... let's remove the test folder if it exists
        shutil.rmtree(IMG_FOLDER_PREFIX + NEW_TEST_LOCATION, ignore_errors=True)

        # make sure that the img is downloaded
        # first test img of riskmap favicon
        def mock_get_image_urls(self):
            return { 0: "https://riskmap.in/assets/logos/url_logo.png"}

        # replace the get_image_urls function with our mock one 
        mut_loader.get_image_urls = types.MethodType(mock_get_image_urls, CognicityImageLoader)
        mut_loader.fetch_images()

        # make sure folder gets created
        self.assertTrue(os.path.exists(IMG_FOLDER_PREFIX + NEW_TEST_LOCATION))

        img_path = IMG_FOLDER_PREFIX+NEW_TEST_LOCATION + "/0.jpeg"
        # make sure the image gets created
        for _, __, files in os.walk(IMG_FOLDER_PREFIX + NEW_TEST_LOCATION):
            self.assertTrue(len(files) > 0)
            for name in files:
                self.assertEqual(name, "0.jpeg")
                self.assertTrue(os.stat(img_path).st_size > 0)

        # clean up after ourselves
        shutil.rmtree(IMG_FOLDER_PREFIX + NEW_TEST_LOCATION, ignore_errors=True)
        pass

if __name__ == "__main__":
    unittest.main()
