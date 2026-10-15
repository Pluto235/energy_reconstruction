import importlib
import unittest
from unittest.mock import patch
import numpy as np
from apply.stages.nonnegative_quadratic import fit_nonnegative, normalized_background

d = importlib.import_module('apply.stages.04_background')

class NonnegativeTests(unittest.TestCase):
    def setUp(self):
        xy = np.arange(-5.95, 6., .1)
        x,y = np.meshgrid(xy,xy)
        r = np.hypot(x,y)
        self.a = d.surface_design_matrix(x[(r>=1.5)&(r<3.5)], y[(r>=1.5)&(r<3.5)], 2)
    def test_feasible_matches_old_ls(self):
        z = self.a @ np.array([3., .01, -.02, .003, .001, .002])
        fit = fit_nonnegative(self.a,z,6.)
        old = np.linalg.lstsq(self.a/np.sqrt(np.maximum(z,1))[:,None],z/np.sqrt(np.maximum(z,1)),rcond=None)[0]
        np.testing.assert_allclose(fit['coefficients'],old,rtol=1e-12,atol=1e-12)
        self.assertFalse(fit['active'])
    def test_negative_ls_is_refit(self):
        z = self.a @ np.array([.1,0,0,-.006,0,-.006])
        fit = fit_nonnegative(self.a,z,6.)
        self.assertTrue(fit['active'])
        self.assertGreaterEqual(d.minimum_quadratic_on_centered_disk(fit['coefficients'],6.),-fit['tolerance'])
        self.assertGreater(fit['objective'],fit['unconstrained_objective'])
    def test_continuous_minima_and_grid_counterexample(self):
        from apply.stages.poisson_roi_background import _minimum_quadratic_on_centered_disk
        for c, minimum in [([1,-2,0,1,0,1],0),([36,0,0,-1,0,-1],0),([5,-2,0,-1,0,0],-43)]:
            self.assertAlmostEqual(d.minimum_quadratic_on_centered_disk(np.array(c),6),minimum,places=9)
            self.assertAlmostEqual(_minimum_quadratic_on_centered_disk(np.array(c),6)[0],minimum,places=9)
        c = np.array([.004,-.1,-.1,1,0,1])
        x,y=np.meshgrid(np.arange(-6,6.01,.1),np.arange(-6,6.01,.1))
        self.assertGreaterEqual(np.min(d.surface_design_matrix(x.ravel(),y.ravel(),2)@c),-1e-14)
        self.assertLess(d.minimum_quadratic_on_centered_disk(c,6),-.0009)
        self.assertLess(_minimum_quadratic_on_centered_disk(c,6)[0],-.0009)
    def test_normalization_and_integral(self):
        c = np.array([3., .01, -.02, .003, .001, .002])
        z = self.a @ c
        fit=fit_nonnegative(self.a,z,6.)
        b,scale=normalized_background(fit,self.a,z,.8,.1)
        self.assertAlmostEqual(scale,1.,places=12)
        from apply.tests.test_background_analytic_bon import gauss_disk_integral
        self.assertAlmostEqual(b,gauss_disk_integral(c,.8,.1,1),places=8)
    def test_failures_never_pass(self):
        with self.assertRaises(ValueError):
            fit_nonnegative(self.a,np.full(len(self.a),np.nan),6.)
        with self.assertRaises(ValueError):
            normalized_background({'coefficients':np.zeros(6),'tolerance':1e-12},self.a,np.zeros(len(self.a)),1.,.1)
        from types import SimpleNamespace
        with patch('apply.stages.nonnegative_quadratic.minimize',return_value=SimpleNamespace(success=False,message='fixture failure')):
            with self.assertRaises(RuntimeError):
                fit_nonnegative(self.a,self.a@np.array([.1,0,0,-.006,0,-.006]),6.)

if __name__=='__main__': unittest.main()

class SavedPathTests(unittest.TestCase):
    def test_historical_relative_paths_are_apply_relative(self):
        from apply.tools.january_v6_nonnegative import saved_path, REPO
        self.assertEqual(saved_path('output/old/file.npz'),REPO/'apply/output/old/file.npz')
        self.assertEqual(saved_path('/absolute/file'),__import__('pathlib').Path('/absolute/file'))

class UncertaintyTests(unittest.TestCase):
    def test_invalid_replicates_remain_in_failure_denominator(self):
        from apply.tools.january_v6_nonnegative import bootstrap, interval
        a=np.column_stack([np.ones(8),np.arange(8),np.zeros(8),np.arange(8)**2,np.zeros(8),np.zeros(8)])
        draws,active,failed=bootstrap(a,np.zeros(8),1.,3,np.random.default_rng(1))
        self.assertEqual(len(failed),3)
        self.assertEqual(len(draws),3)
        self.assertTrue(np.all(np.isnan(draws)))
        self.assertEqual(active,0)
        self.assertEqual(interval(.1,1.,.68)[0],0.)

class EnsembleSummaryTests(unittest.TestCase):
    def test_missing_fits_count_against_coverage(self):
        from apply.tools.january_v6_nonnegative_spectral_toys import summarize
        result=summarize([{'estimate':1.,'error':.1},{'estimate':2.,'error':.1},None],1.)
        self.assertEqual(result['attempted'],3)
        self.assertEqual(result['successful'],2)
        self.assertAlmostEqual(result['coverage68'],1/3)
        self.assertAlmostEqual(result['failure_rate'],1/3)
