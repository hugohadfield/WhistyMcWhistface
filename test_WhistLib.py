import unittest
from alphawhist.core.WhistLib import *

class BasicTests(unittest.TestCase):
    def test_52_cards_in_deck(self):
        assert len(card_to_index_map) == 52


if __name__ == '__main__':
    unittest.main()
