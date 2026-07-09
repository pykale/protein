import unittest

from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor


class MapDiffPredictorTest(unittest.TestCase):
    def test_mapdiff_predictor_generates(self):
        cfg = AutoProteinConfig.from_preset('mapdiff')
        data = AutoProteinPreprocessor.from_config(cfg).transform_sample(
            {'backbone_coords': [[[0, 0, 0]], [[1, 0, 0]]], 'sequence': 'MA'}
        )
        out = AutoProteinPredictor.from_config(cfg).generate(data)
        self.assertTrue('sequences' in out or 'token_ids' in out)


if __name__ == '__main__':
    unittest.main()
