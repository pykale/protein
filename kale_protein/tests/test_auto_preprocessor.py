import unittest

from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor


class AutoProteinPreprocessorTest(unittest.TestCase):
    def test_drugban_preprocess(self):
        cfg = AutoProteinConfig.from_preset('drugban')
        preprocessor = AutoProteinPreprocessor.from_config(cfg)
        out = preprocessor.transform_sample({'smiles': 'CCO', 'sequence': 'MKTFFVLLL', 'label': 1})
        self.assertIn('drug', out)
        self.assertIn('target', out)
        self.assertEqual(out['label'], 1)
        self.assertIn('node_features', out['drug'])
        self.assertIn('tokens', out['target'])

    def test_mapdiff_preprocess(self):
        cfg = AutoProteinConfig.from_preset('mapdiff')
        preprocessor = AutoProteinPreprocessor.from_config(cfg)
        out = preprocessor.transform_sample({'backbone_coords': [[[0, 0, 0]], [[1, 0, 0]]], 'sequence': 'MA'})
        self.assertIn('structure', out)
        self.assertIn('noisy_sequence', out)


if __name__ == '__main__':
    unittest.main()
