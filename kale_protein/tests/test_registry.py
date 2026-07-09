import unittest

from kale_protein.registry.base import Registry


class RegistryTest(unittest.TestCase):
    def test_registry_register_get_available_and_missing(self):
        registry = Registry('thing')

        class A:
            pass

        registry.register('a', A)
        self.assertIs(registry.get('a'), A)
        self.assertEqual(registry.available_keys(), ['a'])
        with self.assertRaises(KeyError):
            registry.get('missing')

    def test_registry_decorator(self):
        registry = Registry('thing')

        @registry.register(('x', 'y'))
        class B:
            pass

        self.assertIs(registry.get(('x', 'y')), B)


if __name__ == '__main__':
    unittest.main()
