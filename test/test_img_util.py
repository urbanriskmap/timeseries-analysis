import unittest

import img_util
import image_recognition.aws_recog


class ImageUtilTest(unittest.TestCase):
    def test_create(self):
        img_util.save_feature_vectors({}, filename="test_save_feat.p")
        self.assertTrue(True)
        return True
