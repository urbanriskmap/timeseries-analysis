import pandas as pd
import requests
import os
from PIL import Image

import numpy as np


class CognicityImageLoader:

    def __init__(self, configObj):
        """ Creates a data loader for cognicity data
        Args:
            configObject(dict)
                databaseEngine: a sqlalchemy database engine
                location: a location string, one of "id" or "ch"
                img_folder_prefix: path for folder to store downloaded images
                logger: a python logging object
        Returns:
            None
        """
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.img_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]
        self.logger.debug("CognicityImageLoader constructed")

    def get_image_urls(self):
        """
        returns dictionary of {pkey: image_url} for
        all rows in db that have an image url
        """
        from sqlalchemy.sql import text

        # TODO: this does an unsafe string concatenation and the
        # configObj.database_name is vulnerable to SQL injection attacks,
        # as such this code should only be run with trusted config files
        # on trusted hardware.
        connection = self.database.connect()
        rows = pd.read_sql_query(
                                text("""
                                SELECT pkey, image_url FROM """ + self.database_name + """.all_reports
                                WHERE image_url IS NOT null
                                ORDER BY created_at
                                """),
                                con=connection,
                                params={"database_name": self.database_name},
                                index_col="pkey")
        return rows.to_dict()["image_url"]

    def __save_image(self, key, raw_bytes):
        try:
            self.logger.debug("Trying to open image")
            im = Image.open(raw_bytes)
            try:
                folder_path = self.img_folder_prefix + self.location
                self.logger.debug("Creating folder: " + folder_path)
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
                              + key + " " + e.strerror)

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
                                     + each + " HTTP STATUS CODE: "
                                     + str(r.status_code))
            except requests.RequestException:
                self.logger.error("Malformed url :" + each)

    def make_matrix_rep(self, feature_dict, len_feat_vect):
        """ Turns a labels dictionary into a matrix

        Args:
            feature_dict ( pkey (int): [  confidence (float)]):
                A dictionary that maps from positive integer
                pkeys into a list of confidence
                values.
            len_feat_vect (int): the number of labels
                (must be same for every pkey)
        Returns:
            matrix_rep (np.array( num_labels+1, num_pkeys)): a matrix
                representation where
                every column is one feature vector
                with the zeroth element being the pkey
                order is preserved in confidece score list,

            Example if pkey 0 has confidece values
            [ 1.12, 2.8984, 3.28227] then
            the zeroth column of matrix_rep, which is matrix_rep[:, 0] is
            np.array([0 (pkey), 1.12, 2.8984, 3.28227]) in order.

            it is up to the caller to maintain which confidence
            score goes with which label

            looks like:
            pkey0    | pkey1 ..
            featvect | featV1

        """
        out = np.zeros((len_feat_vect + 1, len(feature_dict.keys())))
        for i, pkey in enumerate(sorted(feature_dict)):
            assert(pkey >= 0)
            # shallow copy because they're builtins
            confidences = feature_dict[pkey].copy()
            confidences.insert(0, pkey)
            out[:, i] = np.array(confidences)
        return out

    def make_labels_rep(self, featureDict, location="id"):
        # if pkey exists in feature dict, figures out if flooding
        # else zero
        #  start_known_flood = "'2017-02-20 00:00:35.630000-05:00'"
        #  end_known_flood = "'2017-02-23 00:00:35.630000-05:00'"
        if location == "id":
            start_known_flood = "2017-02-20 00:00:35.630000-05:00"
            end_known_flood = "2017-02-23 00:00:35.630000-05:00"
            knownFlood = pd.read_sql_query(
                """
                SELECT pkey from cognicity.all_reports
                WHERE created_at > '2017-02-20 00:00:35.630000-05:00'
                AND created_at < '2017-02-23 00:00:35.630000-05:00'
                """,
                con=self.database,
                params={
                        "start_known_flood": start_known_flood,
                        "end_known_flood": end_known_flood
                       })
        else:
            start_known_flood = "'2017-11-01 00:00:35.630000-04:00'"
            end_known_flood = "'2017-11-07 00:00:35.630000-04:00'"
            knownFlood = pd.read_sql_query(
                """
                SELECT pkey from riskmap.all_reports
                WHERE created_at > %(start_known_flood)s::timestamptz
                AND created_at < %(end_known_flood)s::timestamptz
                """,
                con=self.database,
                params={
                        "start_known_flood": start_known_flood,
                        "end_known_flood": end_known_flood
                       })
        knownFloodSet = set(knownFlood["pkey"])
        self.logger.info("known floodset")
        self.logger.info(knownFloodSet)
        out = np.zeros((2, len(featureDict.keys())))
        for i, pkey in enumerate(sorted(featureDict)):
            # look up if this pkey is a flood event
            if pkey in knownFloodSet:
                out[0, i] = pkey
                out[1, i] = 1
            else:
                # no known flooding
                out[0, i] = pkey
                out[1, i] = -1
        return out
