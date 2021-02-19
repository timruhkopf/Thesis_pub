import unittest
from ..Layer import GAM


class TestGAM(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GAM()
        X, Z, y = self.model.sample_model(n=1000)

    def tearDown(self) -> None:
        pass

    def test_covariance_matrix(self):
        pass

    def test_bijection(self):
        pass

    def test_resample_hierarchical(self):
        """ensure, the a proper variance is updated in W's distribution  """
        # check it does not flatline if variance is not close to zero!
        pass

    def test_update_distribution(self):
        pass

    def test_samplable(self):
        # very simple hierarchical model
        pass


if __name__ == '__main__':
    unittest.main()
