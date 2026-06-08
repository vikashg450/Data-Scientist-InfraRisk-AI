import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

class DegradationPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for structural degradation and Remaining Useful Life (RUL) prediction.
    Models three physics-informed degradation processes:
    1. Paris' Law for fatigue crack propagation: da/dt = C * (Y * d_sigma * sqrt(pi * a))^m * f
    2. AASHTO Pavement Decay: dP/dt = -beta * P * t^alpha
    3. Corrosion Power Law: d(d_c)/dt = A * B * t^(B - 1)
    
    Inputs:
        - t: Time tensor (B, 1), requires_grad=True
    Outputs:
        - a: Crack size (B, 1)
        - P: Present Serviceability Index (B, 1)
        - d_c: Corrosion depth (B, 1)
    """
    def __init__(self, hidden_dim: int = 64):
        super(DegradationPINN, self).__init__()
        
        # Neural network layers mapping time t -> (crack_size, PSI, corrosion_depth)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Outputs: [a_pred, P_pred, dc_pred]
        )
        
        # Scale layers for output scaling to physical ranges
        # crack size a: [0.001 to 0.1] meters (initial to critical)
        # PSI P: [1.5 to 5.0] (terminal to pristine)
        # corrosion depth d_c: [0.0 to 0.05] meters
        
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(t)
        
        # Physical scale transforms
        a = 0.001 + 0.099 * torch.sigmoid(out[:, 0:1])
        P = 1.5 + 3.5 * torch.sigmoid(out[:, 1:2])
        d_c = 0.05 * torch.sigmoid(out[:, 2:3])
        
        return a, P, d_c

    def compute_physics_residuals(
        self, 
        t: torch.Tensor,
        a: torch.Tensor,
        P: torch.Tensor,
        d_c: torch.Tensor,
        physics_params: Dict[str, float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the physics residuals using PyTorch Autograd.
        """
        # Gradients of outputs w.r.t time t
        grad_outputs = torch.ones_like(t)
        
        da_dt = torch.autograd.grad(a, t, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        dP_dt = torch.autograd.grad(P, t, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        ddc_dt = torch.autograd.grad(d_c, t, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        
        # 1. Paris' Law residual: da/dt - C * (Y * d_sigma * sqrt(pi * a))^m * f_cycles
        C = physics_params.get("C", 1.2e-11)
        m = physics_params.get("m", 3.0)
        Y = physics_params.get("Y", 1.12)
        d_sigma = physics_params.get("d_sigma", 150.0) # MPa
        f_cycles = physics_params.get("f_cycles", 1.0e5) # cycles/year
        
        # K_range = Y * d_sigma * sqrt(pi * a)
        # We clamp 'a' to avoid negative square roots
        a_clamped = torch.clamp(a, min=1e-5)
        k_range = Y * d_sigma * torch.sqrt(np.pi * a_clamped)
        da_dt_physics = C * torch.pow(k_range, m) * f_cycles
        residual_paris = da_dt - da_dt_physics
        
        # 2. AASHTO Pavement Decay residual: dP/dt + beta * P * t^alpha
        beta = physics_params.get("beta", 0.05)
        alpha = physics_params.get("alpha", 0.8)
        # Avoid issues at t = 0
        t_clamped = torch.clamp(t, min=1e-5)
        dP_dt_physics = -beta * P * torch.pow(t_clamped, alpha)
        residual_aashto = dP_dt - dP_dt_physics
        
        # 3. Corrosion Power Law residual: d(d_c)/dt - A * B * t^(B - 1)
        corr_A = physics_params.get("corr_A", 0.002)
        corr_B = physics_params.get("corr_B", 0.5)
        ddc_dt_physics = corr_A * corr_B * torch.pow(t_clamped, corr_B - 1.0)
        residual_corrosion = ddc_dt - ddc_dt_physics
        
        return residual_paris, residual_aashto, residual_corrosion

    def fit(
        self,
        t_data: np.ndarray,
        a_data: np.ndarray,
        P_data: np.ndarray,
        dc_data: np.ndarray,
        physics_params: Dict[str, float],
        epochs: int = 500,
        lr: float = 1e-3,
        lambda_phys: float = 0.1
    ):
        """Trains the network using a combined loss (data loss + physics residuals loss)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Move inputs to PyTorch tensors and enable grad on time tensor for PINN physics loss
        t_t = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        a_t = torch.tensor(a_data, dtype=torch.float32).view(-1, 1)
        P_t = torch.tensor(P_data, dtype=torch.float32).view(-1, 1)
        dc_t = torch.tensor(dc_data, dtype=torch.float32).view(-1, 1)
        
        # For physics collocation points, let's create a range of times from 0 to max_time
        max_time = float(np.max(t_data))
        t_colloc = torch.linspace(0.01, max_time * 1.5, steps=200, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. Data Loss (supervised)
            a_pred, P_pred, dc_pred = self.forward(t_t)
            loss_data = F.mse_loss(a_pred, a_t) + F.mse_loss(P_pred, P_t) + F.mse_loss(dc_pred, dc_t)
            
            # 2. Physics Collocation Loss
            a_col, P_col, dc_col = self.forward(t_colloc)
            res_paris, res_aashto, res_corrosion = self.compute_physics_residuals(
                t_colloc, a_col, P_col, dc_col, physics_params
            )
            loss_physics = torch.mean(res_paris**2) + torch.mean(res_aashto**2) + torch.mean(res_corrosion**2)
            
            # Total loss
            loss = loss_data + lambda_phys * loss_physics
            loss.backward()
            optimizer.step()
            
    def predict_rul(self, current_t: float, physics_params: Dict[str, float], time_step: float = 0.5, max_years: float = 100.0) -> Tuple[float, str]:
        """
        Estimates the Remaining Useful Life (RUL) by stepping forward in time
        until any of the degradation variables hit their critical threshold:
            - Crack size (a) > 0.05 meters (50 mm)
            - Present Serviceability Index (P) < 2.0 (critical pavement failure)
            - Corrosion depth (d_c) > 0.02 meters (20 mm)
        Returns:
            rul_years: Remaining useful life in years
            limiting_factor: The failure mode that triggered the limit
        """
        self.eval()
        t = current_t
        a_limit = 0.05
        P_limit = 2.0
        dc_limit = 0.02
        
        with torch.no_grad():
            while t < current_t + max_years:
                t_tensor = torch.tensor([[t]], dtype=torch.float32)
                a, P, d_c = self.forward(t_tensor)
                
                a_val = a.item()
                P_val = P.item()
                dc_val = d_c.item()
                
                if a_val >= a_limit:
                    return max(0.0, t - current_t), "Fatigue Crack Limit (Paris' Law)"
                if P_val <= P_limit:
                    return max(0.0, t - current_t), "Pavement Roughness Limit (AASHTO)"
                if dc_val >= dc_limit:
                    return max(0.0, t - current_t), "Steel Corrosion Limit"
                    
                t += time_step
                
        return max_years, "No limit reached within maximum horizon"

    def predict_climate_adjusted_rul(
        self,
        current_t: float,
        physics_params: Dict[str, float],
        scenario: str = "RCP8.5",
        time_step: float = 0.5,
        max_years: float = 100.0
    ) -> Tuple[float, str]:
        """
        Estimates the Remaining Useful Life (RUL) under climate scenarios (e.g. IPCC RCP 8.5)
        by applying physical acceleration factors to time in the neural network forward pass:
            - RCP8.5 accelerates structural degradation: crack growth by 1.15x, pavement decay by 1.4x, corrosion by 1.3x.
        """
        self.eval()
        t = current_t
        a_limit = 0.05
        P_limit = 2.0
        dc_limit = 0.02
        
        # Acceleration factors
        if scenario == "RCP8.5":
            accel_a = 1.15
            accel_P = 1.40
            accel_dc = 1.30
        elif scenario == "RCP4.5":
            accel_a = 1.05
            accel_P = 1.15
            accel_dc = 1.10
        else:
            accel_a = 1.0
            accel_P = 1.0
            accel_dc = 1.0
            
        with torch.no_grad():
            while t < current_t + max_years:
                # Query forward at effective time steps for each failure mode
                t_tensor_a = torch.tensor([[current_t + (t - current_t) * accel_a]], dtype=torch.float32)
                t_tensor_P = torch.tensor([[current_t + (t - current_t) * accel_P]], dtype=torch.float32)
                t_tensor_dc = torch.tensor([[current_t + (t - current_t) * accel_dc]], dtype=torch.float32)
                
                a, _, _ = self.forward(t_tensor_a)
                _, P, _ = self.forward(t_tensor_P)
                _, _, d_c = self.forward(t_tensor_dc)
                
                a_val = a.item()
                P_val = P.item()
                dc_val = d_c.item()
                
                if a_val >= a_limit:
                    return max(0.0, t - current_t), f"Fatigue Crack Limit (Paris' Law) under {scenario}"
                if P_val <= P_limit:
                    return max(0.0, t - current_t), f"Pavement Roughness Limit (AASHTO) under {scenario}"
                if dc_val >= dc_limit:
                    return max(0.0, t - current_t), f"Steel Corrosion Limit under {scenario}"
                    
                t += time_step
                
        return max_years, f"No limit reached within maximum horizon under {scenario}"

