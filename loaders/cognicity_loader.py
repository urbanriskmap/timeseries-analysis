import pandas as pd
import requests
import os
from PIL import Image

import numpy as np


class CognicityLoader:
    """ Loads data from the cognicity database defined in
        config constructor argument

    """

    def __init__(self, configObj):
        """ Creates a data loader for cognicity data
        Args:
            configObject(dict)
                databaseEngine: a sqlalchemy database engine
                location: a location string, one of "id" or "ch"
                data_folder_prefix: path for folder to store downloaded images
                logger: a python logging object
        Returns:
            None
        """
        self.config = configObj
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.data_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]
        self.logger.debug("CognicityLoader constructed")
        if not os.path.exists(self.data_folder_prefix):
            os.makedirs(self.data_folder_prefix)
            self.logger.debug(
                "data folder doesn't exist, created path:" +
                self.data_folder_prefix)

    def get_image_urls(self):
        """
        returns dictionary of {pkey: image_url} for
        all rows in db that have an image url
        """

        # TODO: this does an unsafe string concatenation and the
        # configObj.database_name is vulnerable to SQL injection attacks,
        # as such this code should only be run with trusted config files
        # on trusted hardware.
        connection = self.database.connect()
        rows = pd.read_sql_query(
                                """
                                SELECT pkey, image_url FROM """ + self.database_name + """.all_reports
                                WHERE image_url IS NOT null
                                ORDER BY created_at
                                """,
                                con=connection,
                                params={"database_name": self.database_name},
                                index_col="pkey")
        return rows.to_dict()["image_url"]

    def get_texts(self):
        """
        Returns:
            pandas dataframe of pkey to all the reports in the database
        """
        rows = pd.read_sql_query('''
        SELECT pkey, text from ''' + self.database_name + '''.all_reports
            WHERE text IS NOT null
            AND LENGTH(text) > 0
            AND text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%'
            ORDER BY created_at
        ''', con=self.database, index_col="pkey")
        return rows

    def get_flood_depths(self):
        pkeys_and_flood_depth = pd.read_sql_query("""
        SELECT pkey, CAST(report_data ->> 'flood_depth' AS INTEGER)
            AS flood_depth
            FROM """ + self.database_name + """.all_reports
        WHERE report_data->>'flood_depth' IS NOT NULL
        """, con=self.database, index_col="pkey")

        # make sure all pkeys are in here, else write zero
        included = pkeys_and_flood_depth.index
        all_pkeys = pd.Index(self.config["all_pkeys"])
        # in all pkeys but not in the ones we just queired
        diff = all_pkeys.difference(included)

        np_zeros = np.array([0]*diff.size)
        zero_flood_depth = pd.DataFrame(data=np_zeros,
                                        index=diff,
                                        columns=["flood_depth"])
        zero_flood_depth = zero_flood_depth.rename_axis(index="pkey")
        pkeys_and_flood_depth = pkeys_and_flood_depth.append(zero_flood_depth)

        return pkeys_and_flood_depth

    def __save_image(self, key, raw_bytes):
        try:
            self.logger.debug("Trying to open image")
            im = Image.open(raw_bytes)
            try:
                folder_path = os.path.join(self.data_folder_prefix, self.location)
                self.logger.debug("Creating img folder: " + folder_path)
                os.mkdir(folder_path)
            except FileExistsError:
                self.logger.debug(
                        "Tried to create an image"
                        "folder that already existed, continuing")
            try:
                path = folder_path + "/" + str(key) + ".jpeg"
                self.logger.debug("image mode: " + im.mode)
                im = im.convert("RGB")
                im.save(path, "JPEG")
            except IOError as e:
                self.logger.error("IOError: Unable to write file to path: "
                                  + path + e.strerror)
            except KeyError:
                self.logger.error("KeyError: Unable to write file to path:"
                                  + path)
        except IOError as e:
            self.logger.error("Unable to open image with pkey: "
                              + str(key) + " " + e.strerror)

    def fetch_images(self):

        image_urls = self.get_image_urls()
        for key in image_urls.keys():
            each = image_urls[key]
            self.logger.info("Image Url is: " + each)
            try:
                r = requests.get(each, stream=True)
                if r.status_code == 200:
                    self.__save_image(key, r.raw)
                else:
                    self.logger.info("ERROR COULD NOT READ URL: "
                                     + str(each) + " HTTP STATUS CODE: "
                                     + str(r.status_code))
            except requests.RequestException:
                self.logger.error("Malformed url :" + each)
