import unittest
from unittest.mock import MagicMock

# Import modules under test
from app import round_sig, classify, make_size_key, build_bom, SolidRow, BomRow

class TestAppCore(unittest.TestCase):
    def test_round_sig(self):
        self.assertEqual(round_sig(10.123, 0.25), 10.0)
        self.assertEqual(round_sig(10.22, 0.25), 10.25)
        self.assertEqual(round_sig(10.123, 0.0), 10.123)
        self.assertEqual(round_sig(10.123, -1.0), 10.123)

    def test_classify(self):
        # plate: T < 0.2 * W and T < 0.1 * L
        # L=100, W=50, T=4
        self.assertEqual(classify(100.0, 50.0, 4.0), "plate")
        
        # pin: W ~ T and L / T > 6.0
        # L=100, W=10, T=10
        self.assertEqual(classify(100.0, 10.0, 10.0), "pin")
        
        # profile: otherwise
        # L=100, W=30, T=20
        self.assertEqual(classify(100.0, 30.0, 20.0), "profile")

    def test_make_size_key(self):
        # plate
        self.assertEqual(make_size_key("plate", 100.0, 50.0, 5.0), "100.0×50.0×T5.0 mm")
        # pin
        self.assertEqual(make_size_key("pin", 100.0, 10.0, 10.0), "Ø10.0×100.0 mm")
        # profile
        self.assertEqual(make_size_key("profile", 100.0, 30.0, 20.0), "L100.0 W30.0 T20.0 mm")

    def test_build_bom(self):
        # Create a list of SolidRows
        # Let's create two identical plates (sig='sig_plate') and one profile (sig='sig_prof')
        s1 = SolidRow(idx=1, cls="plate", name="Plate A", L_mm=100.0, W_mm=50.0, T_mm=5.0, Vol_cm3=25.0, Area_cm2=100.0, Weight_kg=0.2, sig="sig_plate")
        s2 = SolidRow(idx=2, cls="plate", name="Plate B", L_mm=100.0, W_mm=50.0, T_mm=5.0, Vol_cm3=25.0, Area_cm2=100.0, Weight_kg=0.2, sig="sig_plate")
        s3 = SolidRow(idx=3, cls="profile", name="Prof A", L_mm=150.0, W_mm=30.0, T_mm=20.0, Vol_cm3=90.0, Area_cm2=200.0, Weight_kg=0.7, sig="sig_prof")

        solids = [s1, s2, s3]
        bom = build_bom(solids)

        # Output should be sorted by class rank ("profile": 0, "plate": 1, "pin": 2)
        # So profile (POS 1) comes first, then plate (POS 2)
        self.assertEqual(len(bom), 2)
        
        # Profile checks
        self.assertEqual(bom[0].pos, 1)
        self.assertEqual(bom[0].class_name, "profile")
        self.assertEqual(bom[0].qty, 1)
        self.assertEqual(bom[0].avg_weight_kg, 0.7)
        self.assertEqual(bom[0].total_weight_kg, 0.7)

        # Plate checks
        self.assertEqual(bom[1].pos, 2)
        self.assertEqual(bom[1].class_name, "plate")
        self.assertEqual(bom[1].qty, 2)
        self.assertEqual(bom[1].avg_weight_kg, 0.2)
        self.assertEqual(bom[1].total_weight_kg, 0.4)
        # Verify names are aggregated
        self.assertEqual(bom[1].names, "Plate A, Plate B")

if __name__ == "__main__":
    unittest.main()
