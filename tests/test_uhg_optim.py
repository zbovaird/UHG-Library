"""
Tests for UHG optimizers.
"""

import torch
import pytest
from uhg.optim import UHGBaseOptimizer, UHGAdam, UHGSGD
from uhg.metrics import UHGMetric

def create_uhg_parameters(dim: int = 3) -> torch.Tensor:
    """Create random parameters on the UHG manifold."""
    param = torch.randn(dim, requires_grad=True)
    norm = torch.norm(param, p=2)
    return (param / (norm + 1e-6)).clone().detach().requires_grad_(True)

def test_base_optimizer_initialization():
    """Test UHGBaseOptimizer initialization."""
    params = [create_uhg_parameters()]
    optimizer = UHGBaseOptimizer(params, {'lr': 0.1, 'eps': 1e-8})
    
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert optimizer.param_groups[0]['eps'] == 1e-8
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        UHGBaseOptimizer(params, {'lr': -0.1})
    
    # Test invalid epsilon
    with pytest.raises(ValueError):
        UHGBaseOptimizer(params, {'eps': -1e-8})

def test_manifold_projection():
    """Test manifold projection preserves geometric properties."""
    params = [create_uhg_parameters()]
    optimizer = UHGBaseOptimizer(params, {'lr': 0.1})
    
    # Test projection of random vector
    random_vec = torch.randn(3)
    projected = optimizer._project_to_manifold(random_vec, optimizer.param_groups[0])
    
    # Check nonzero norm (projective geometry)
    norm = torch.norm(projected, p=2)
    assert norm > optimizer.param_groups[0].get('eps', 1e-8)
    
    # Check projection preserves direction
    original_dir = random_vec / (torch.norm(random_vec, p=2) + 1e-6)
    projected_dir = projected / (torch.norm(projected, p=2) + 1e-6)
    assert torch.allclose(original_dir, projected_dir, atol=1e-4)

def test_hyperbolic_gradient():
    """Test hyperbolic gradient computation preserves tangent space structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGBaseOptimizer(params, {'lr': 0.1, 'eps': 1e-8})
    metric = UHGMetric()
    
    # Create random gradient
    grad = torch.randn(3)
    param = params[0]
    
    # Compute hyperbolic gradient
    hgrad = optimizer._compute_hyperbolic_gradient(grad, param, optimizer.param_groups[0])
    
    # Check gradient is in tangent space (orthogonal to parameter)
    dot_product = torch.dot(hgrad, param)
    assert abs(dot_product) < 1e-4  # Should be approximately zero
    
    # Check gradient preserves hyperbolic structure
    metric_tensor = metric.get_metric_tensor(param)
    assert torch.allclose(
        torch.matmul(metric_tensor, hgrad),
        hgrad,
        atol=1e-4
    )

def test_optimizer_step():
    """Test single optimization step preserves manifold structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGBaseOptimizer(params, {'lr': 0.1})
    
    # Create dummy loss
    loss = torch.sum(params[0])
    loss.backward()
    
    # Take optimization step
    optimizer.step()
    
    # Check parameters still satisfy manifold constraints
    assert optimizer._check_manifold_constraint(params[0], optimizer.param_groups[0])

def test_uhg_adam_initialization():
    """Test UHGAdam initialization."""
    params = [create_uhg_parameters()]
    optimizer = UHGAdam(params, lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
    assert optimizer.param_groups[0]['eps'] == 1e-8
    
    # Test invalid betas
    with pytest.raises(ValueError):
        UHGAdam(params, betas=(-0.1, 0.999))
    with pytest.raises(ValueError):
        UHGAdam(params, betas=(0.9, 1.1))

def test_uhg_adam_momentum():
    """Test UHGAdam momentum updates preserve hyperbolic structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGAdam(params, lr=0.1)
    
    # Create random gradient
    grad = torch.randn(3)
    param = params[0]
    
    # Compute momentum update
    beta1 = optimizer.param_groups[0]['betas'][0]
    momentum = optimizer._compute_hyperbolic_momentum(grad, param, beta1, optimizer.param_groups[0])
    
    # Check momentum is in tangent space
    dot_product = torch.dot(momentum, param)
    assert abs(dot_product) < 1e-4

def test_uhg_adam_second_moment():
    """Test UHGAdam second moment updates preserve hyperbolic structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGAdam(params, lr=0.1)
    
    # Create random gradient
    grad = torch.randn(3)
    param = params[0]
    
    # Compute second moment update
    beta2 = optimizer.param_groups[0]['betas'][1]
    second_moment = optimizer._compute_hyperbolic_second_moment(grad, param, beta2, optimizer.param_groups[0])
    
    # Check second moment is in tangent space
    dot_product = torch.dot(second_moment, param)
    assert abs(dot_product) < 1e-4

def test_uhg_adam_step():
    """Test UHGAdam optimization step preserves manifold structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGAdam(params, lr=0.1)
    
    # Create dummy loss
    loss = torch.sum(params[0])
    loss.backward()
    
    # Take optimization step
    optimizer.step()
    
    # Check parameters still satisfy manifold constraints
    assert optimizer._check_manifold_constraint(params[0], optimizer.param_groups[0])

def test_uhg_sgd_initialization():
    """Test UHGSGD initialization."""
    params = [create_uhg_parameters()]
    optimizer = UHGSGD(params, lr=0.1, momentum=0.9, weight_decay=0.0)
    
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert optimizer.param_groups[0]['momentum'] == 0.9
    assert optimizer.param_groups[0]['weight_decay'] == 0.0
    
    # Test invalid momentum
    with pytest.raises(ValueError):
        UHGSGD(params, momentum=-0.1, lr=0.1)
    
    # Test invalid weight decay
    with pytest.raises(ValueError):
        UHGSGD(params, lr=0.1, weight_decay=-0.1)

def test_uhg_sgd_momentum():
    """Test UHGSGD momentum updates preserve hyperbolic structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGSGD(params, lr=0.1, momentum=0.9)
    
    # Create random gradient
    grad = torch.randn(3)
    param = params[0]
    
    # Compute momentum update
    momentum = optimizer._compute_hyperbolic_momentum(grad, param, optimizer.param_groups[0])
    
    # Check momentum is in tangent space
    dot_product = torch.dot(momentum, param)
    assert abs(dot_product) < 1e-4

def test_uhg_sgd_step():
    """Test UHGSGD optimization step preserves manifold structure."""
    params = [create_uhg_parameters()]
    optimizer = UHGSGD(params, lr=0.1, momentum=0.9)
    
    # Create dummy loss
    loss = torch.sum(params[0])
    loss.backward()
    
    # Take optimization step
    optimizer.step()
    
    # Check parameters still satisfy manifold constraints
    assert optimizer._check_manifold_constraint(params[0], optimizer.param_groups[0])

def test_optimizer_convergence():
    """Test optimizer convergence to a target point in hyperbolic space."""
    # Create initial parameters
    initial_params = [create_uhg_parameters()]
    target_params = [create_uhg_parameters()]

    # Create optimizer with momentum for better convergence
    optimizer = UHGSGD(initial_params, lr=0.01, momentum=0.9)
    metric = UHGMetric()

    # Define hyperbolic distance loss
    def hyperbolic_distance_loss():
        return metric.hyperbolic_distance(initial_params[0], target_params[0])

    # Optimize with learning rate decay
    for step in range(1000):
        optimizer.zero_grad()
        loss = hyperbolic_distance_loss()
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(initial_params, max_norm=0.1)

        # Decay learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01 * (1.0 - step / 1000)

        optimizer.step()

    # Check convergence using hyperbolic distance
    final_distance = hyperbolic_distance_loss()
    assert final_distance < 0.1  # Should be close to target 