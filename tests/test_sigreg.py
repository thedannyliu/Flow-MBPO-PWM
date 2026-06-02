import torch

from flow_mbpo_pwm.utils.sigreg import SIGRegLoss, add_sigreg_loss, latent_gaussian_stats


def test_sigreg_loss_is_finite_for_random_embeddings():
    torch.manual_seed(0)
    embeddings = torch.randn(4, 8, 6)
    loss = SIGRegLoss(num_knots=5, num_projections=16)(embeddings)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sigreg_loss_backpropagates_to_embeddings():
    torch.manual_seed(1)
    embeddings = torch.randn(3, 7, 5, requires_grad=True)
    loss = SIGRegLoss(num_knots=5, num_projections=12)(embeddings)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_constant_embeddings_receive_nonzero_penalty():
    embeddings = torch.zeros(3, 8, 5)
    loss = SIGRegLoss(num_knots=5, num_projections=12)(embeddings)
    assert loss.item() > 0.0


def test_zero_weight_is_noop_for_loss_and_gradients():
    torch.manual_seed(2)
    embeddings_a = torch.randn(3, 6, 4, requires_grad=True)
    embeddings_b = embeddings_a.detach().clone().requires_grad_(True)
    base_a = embeddings_a.square().mean()
    base_b = embeddings_b.square().mean()

    total, reg = add_sigreg_loss(
        base_a,
        embeddings_a,
        weight=0.0,
        regularizer=SIGRegLoss(num_knots=5, num_projections=12),
    )
    total.backward()
    base_b.backward()

    assert reg.item() == 0.0
    assert torch.equal(total.detach(), base_a.detach())
    assert torch.allclose(embeddings_a.grad, embeddings_b.grad)


def test_latent_gaussian_stats_are_finite_and_include_isotropy_proxy():
    torch.manual_seed(3)
    embeddings = torch.randn(5, 9, 7)
    stats = latent_gaussian_stats(embeddings)
    values = [
        stats.mean_norm,
        stats.variance_mean,
        stats.variance_min,
        stats.variance_max,
        stats.covariance_offdiag_abs_mean,
    ]
    assert all(torch.isfinite(value) for value in values)
    assert stats.variance_mean.item() > 0.0
    assert stats.covariance_offdiag_abs_mean.item() >= 0.0
