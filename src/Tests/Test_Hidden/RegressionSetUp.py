import torch
import torch.nn as nn

from src.Layer.Hidden import Hidden


class RegressionSetUp:
    def setUp(self) -> None:
        self.steps = 1000
        # Regression example data (from Hidden layer)
        # TODO make it 5 regressors + bias!
        p = 1  # no. regressors
        self.model = Hidden(p, 1, bias=True, activation=nn.Identity())
        X, y = self.model.sample_model(n=1000)
        self.model.reset_parameters()
        self.X = X
        self.y = y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.get_device_name(0)
        self.X.to(device)
        self.y.to(device)

        X = self.X.clone()
        X = torch.cat([torch.ones(X.shape[0], 1), X], 1)
        self.model.LS = torch.inverse(X.t() @ X) @ X.t() @ y  # least squares
        self.model_in_LS_format = lambda: torch.cat([self.model.b.reshape((1, 1)), self.model.W.data], 0)
