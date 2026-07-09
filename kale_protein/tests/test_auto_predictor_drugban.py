import unittest

from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor


class DrugBANPredictorTest(unittest.TestCase):
    def test_drugban_predictor_predicts(self):
        cfg = AutoProteinConfig.from_preset('drugban')
        data = AutoProteinPreprocessor.from_config(cfg).transform_sample(
            {'smiles': 'CCO', 'sequence': 'MKTFFVLLL', 'label': 1}
        )
        out = AutoProteinPredictor.from_config(cfg).predict(data)
        self.assertIn('logits', out)
        self.assertIn('probabilities', out)


if __name__ == '__main__':
    unittest.main()
