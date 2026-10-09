import unittest

from bom_builder import build_bom
from cad_helpers import SolidRow, classify, make_size_key, round_sig

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

        # pin: W ~ T and long L
        # L=100, W=10, T=10
        self.assertEqual(classify(100.0, 10.0, 10.0), "pin")

        # profile: otherwise
        # L=100, W=30, T=20
        self.assertEqual(classify(100.0, 30.0, 20.0), "profile")

        # extreme/invalid inputs
        self.assertEqual(classify(0.0, 10.0, 5.0), "profile")
        self.assertEqual(classify(-10.0, 5.0, 1.0), "profile")
        self.assertEqual(classify("invalid", 5.0, 1.0), "profile")
        self.assertEqual(classify(100.0, None, 1.0), "profile")


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

    def test_build_bom_empty(self):
        bom = build_bom([])
        self.assertEqual(bom, [])

    def test_build_bom_sorting(self):
        # profile (rank 0), plate (rank 1), pin (rank 2)
        s_pin = SolidRow(idx=1, cls="pin", name="Pin A", L_mm=100.0, W_mm=10.0, T_mm=10.0, Vol_cm3=8.0, Area_cm2=30.0, Weight_kg=0.06, sig="sig_pin")
        s_plate = SolidRow(idx=2, cls="plate", name="Plate A", L_mm=80.0, W_mm=50.0, T_mm=5.0, Vol_cm3=20.0, Area_cm2=80.0, Weight_kg=0.16, sig="sig_plate")
        s_profile = SolidRow(idx=3, cls="profile", name="Prof A", L_mm=120.0, W_mm=20.0, T_mm=20.0, Vol_cm3=48.0, Area_cm2=100.0, Weight_kg=0.38, sig="sig_prof")

        bom = build_bom([s_pin, s_plate, s_profile])
        self.assertEqual(len(bom), 3)
        self.assertEqual(bom[0].class_name, "profile")
        self.assertEqual(bom[1].class_name, "plate")
        self.assertEqual(bom[2].class_name, "pin")

    def test_materials_and_density(self):
        from cad_helpers import convert_density, get_material_density, list_supported_materials

        self.assertEqual(get_material_density("Structural Steel (S235/S355)"), 7850.0)
        self.assertEqual(get_material_density("aluminum"), 2700.0)
        self.assertEqual(get_material_density("unknown_mat", fallback=5000.0), 5000.0)
        self.assertIn("Structural Steel (S235/S355)", list_supported_materials())

        # Density conversion: 7850 kg/m3 = 7.85 g/cm3
        self.assertAlmostEqual(convert_density(7850.0, "kg/m3", "g/cm3"), 7.85, places=2)
        self.assertAlmostEqual(convert_density(7.85, "g/cm3", "kg/m3"), 7850.0, places=1)

    def test_aspect_ratios_and_slenderness(self):
        from cad_helpers import aspect_ratios, compute_scrap_percentage, estimate_raw_stock_weight, slenderness_ratio

        lw, wt = aspect_ratios(100.0, 50.0, 10.0)
        self.assertEqual(lw, 2.0)
        self.assertEqual(wt, 5.0)

        slenderness = slenderness_ratio(100.0, 10.0, 5.0)
        self.assertEqual(slenderness, 20.0)

        # 100 x 50 x 10 mm = 50,000 mm3 = 5e-5 m3 * 7850 = 0.3925 kg
        stock_w = estimate_raw_stock_weight(100.0, 50.0, 10.0, 7850.0)
        self.assertAlmostEqual(stock_w, 0.3925, places=4)

        scrap = compute_scrap_percentage(0.200, stock_w)
        self.assertGreater(scrap, 0.0)
        self.assertLess(scrap, 100.0)
        self.assertEqual(compute_scrap_percentage(0.5, 0.4), 0.0)


if __name__ == "__main__":
    unittest.main()

