import os


class AbstractLoader:
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
        # make the img directory as well

        folder_path = os.path.join(self.data_folder_prefix, self.location)
        self.image_path = folder_path
        if not os.path.exists(self.image_path):
            self.logger.debug("Creating img folder: " + folder_path)
            os.mkdir(self.image_path)

    def get_image_urls(self):
        """
        returns dictionary of {pkey: image_url} for
        all rows in db that have an image url
        """
        pass

    def get_texts(self):
        """
        Returns:
            pandas dataframe of pkey to all the reports in the database
        """
        pass

    def get_flood_depths(self):
        pass

    def __save_image(self, key, raw_bytes):
        pass

    def fetch_images(self):
        pass
