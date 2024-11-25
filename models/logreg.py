import torch
import torch.nn as nn

_CPU_DEVICE = torch.device("cpu")
DEFAULT_MAX_ITER = 1_000

try:
    import cupy
    from cuml.linear_model import LogisticRegression
    CUML_AVAILABLE = True
except ModuleNotFoundError:
    CUML_AVAILABLE = False
    print('cuML not found, using sklearn')
    from sklearn.linear_model import LogisticRegression

class LogRegModule(nn.Module):
    def __init__(
        self,
        C,
        max_iter=DEFAULT_MAX_ITER,
        dtype=torch.float64,
        device=_CPU_DEVICE,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device if CUML_AVAILABLE else _CPU_DEVICE
        self.estimator = LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            tol=1e-12,
        )

    def forward(self, samples, targets):
        samples_device = samples.device
        samples = samples.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            samples = samples.numpy()
        probas = self.estimator.predict_proba(samples)
        probas = probas if not CUML_AVAILABLE else cupy.asnumpy(probas)
        return {"preds": torch.from_numpy(probas).to(samples_device), "target": targets}

    def fit(self, train_features, train_labels):
        train_features = train_features.to(dtype=self.dtype, device=self.device)
        train_labels = train_labels.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            # both cuML and sklearn only work with numpy arrays on CPU
            train_features = train_features.numpy()
            train_labels = train_labels.numpy()
        self.estimator.fit(train_features, train_labels)
