import unittest

import numpy as np

from lotto_engine.v31_pair_residual_audit import PAIR_INCIDENCE, residualize_pair_edges


class V31PairResidualAuditTests(unittest.TestCase):
    def test_pure_vertex_main_effects_residualize_to_zero(self):
        vertex = np.linspace(-2.0, 3.0, 45)
        edges = PAIR_INCIDENCE @ vertex
        residual, diagnostics = residualize_pair_edges(edges)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-10))
        self.assertAlmostEqual(diagnostics["explained_variance_fraction"], 1.0, places=10)

    def test_residual_is_orthogonal_to_vertex_design(self):
        rng = np.random.default_rng(31)
        edges = rng.normal(size=PAIR_INCIDENCE.shape[0])
        residual, _ = residualize_pair_edges(edges)
        normal_equation_residual = PAIR_INCIDENCE.T @ residual
        self.assertLess(float(np.max(np.abs(normal_equation_residual))), 1e-10)

    def test_pair_incidence_has_two_vertices_per_edge(self):
        self.assertEqual(PAIR_INCIDENCE.shape, (990, 45))
        self.assertTrue(np.all(np.sum(PAIR_INCIDENCE, axis=1) == 2.0))


if __name__ == "__main__":
    unittest.main()
