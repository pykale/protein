import unittest

from kale_protein.auto import AutoProteinConfig, StreamSpec


class AutoProteinConfigTest(unittest.TestCase):
    def test_load_presets_and_streams(self):
        drugban = AutoProteinConfig.from_preset('drugban')
        mapdiff = AutoProteinConfig.from_preset('mapdiff')
        self.assertEqual(drugban['task'], 'drug_target_interaction')
        self.assertEqual(mapdiff['runner'], 'diffusion_generate')
        self.assertIsInstance(drugban.get_streams()['drug'], StreamSpec)

    def test_load_yaml_file(self):
        config = AutoProteinConfig.from_yaml('kale_protein/presets/drugban.yaml')
        self.assertEqual(config['name'], 'drugban')
        self.assertEqual(config.get_streams()['target'].processor, 'amino_acid_tokenizer')

    def test_validate_required_fields(self):
        with self.assertRaises(ValueError):
            AutoProteinConfig.from_dict({'task': 'x'})


if __name__ == '__main__':
    unittest.main()
