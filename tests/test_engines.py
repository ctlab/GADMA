import unittest

from gadma import *

class TestEngines(unittest.TestCase):
    def test_existence(self):
        self.assertTrue(len(list(all_engines())) > 0)
        for engine in all_engines():
            options = {}
            if engine.id == 'dadi':
                options['pts'] = [10,20,30]
            with self.assertRaises(ValueError):
                engine.evaluate([], **options)

