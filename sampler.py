from scipy import integrate
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_intermediate(intermediate_samples, device='cuda', dtype=torch.float32):
    # 8 images to visualize
    num_visualization_timesteps = 8
    # selects evenly spaces images
    visualization_indices = np.linspace(0, len(intermediate_samples) - 1, num_visualization_timesteps, dtype=int)
    visualization_samples = [intermediate_samples[i] for i in visualization_indices]

    fig, axes = plt.subplots(1, 8, figsize=(20, 2.5))
    for i, ax in enumerate(axes):
        sample = torch.tensor(visualization_samples[i], device=device, dtype=dtype)
        sample = sample.clamp(0.0, 1.0)
        # necessary transform for cifar10
        sample = sample[0].permute(2, 1, 0).squeeze(0)
        # print(sample.shape)
        ax.imshow(sample.cpu().numpy(), vmin=0., vmax=1.)
        ax.axis('off')
    plt.show()

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3,
                dtype=torch.float32,
                visualize=True):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  start_time = time.time()
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 3, 32, 32, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps, dtype=dtype):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=dtype).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=dtype).reshape((sample.shape[0], ))

    with torch.no_grad():    
        # get score estimate to be integrated
        score = score_model(sample, time_steps)
    score_np = score.cpu().numpy().reshape((-1,)).astype(np.float32)  # Adjusted to support mps tensors

    return score_np
  
  intermediate_samples = []
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    if visualize:
        intermediate_samples.append(torch.tensor(x, device=device, dtype=dtype).reshape(shape))
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
       
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device, dtype=dtype).reshape(shape)

  if visualize:
       plot_intermediate(intermediate_samples, device, dtype)

  end_time = time.time()
  print("Elapsed sample time:", end_time - start_time)
  return x