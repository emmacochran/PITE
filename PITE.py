import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def conditional_inequality_loss(y_pred, P_time, threshold_p_scaled, max_allowed=0.96):
    """
    Only penalize predictions over max_allowed value when P_F > 0,
    no penalty for times without precipitation
    """
    # Mask: True where constraint applies (P_time < 20)
    constraint_applies = P_time > threshold_p_scaled*-1
    
    # Only get predictions where constraint applies
    y_constrained = y_pred[constraint_applies]
    
    if y_constrained.shape[0] == 0:  # No points where constraint applies
        return torch.tensor(0.0, device=y_pred.device)
    
    # Penalize if prediction is too close to forbidden value (1.0)
    violation = torch.relu(y_constrained - max_allowed)  # Positive when too close
    
    return torch.mean(violation**2)

def calculate_loss(y_hotanddry_pred, y_fallow_pred, pde_residual,
                   y_hotanddry_target, y_fallow_target, y_pde_pred, lambda_ineq,
                   threshold_p_scaled, P_time_pde=None):
    """
    Calculating boundary conditions losses with mean squared error
    then adding all losses together for total_loss
    """
    loss_hotanddry = torch.mean((y_hotanddry_pred - y_hotanddry_target)**2)
    loss_fallow    = torch.mean((y_fallow_pred - y_fallow_target)**2)
    loss_pde       = torch.mean(pde_residual**2)
    
    loss_ineq = torch.tensor(0.0, device=pde_residual.device)
    if P_time_pde is not None:
        loss_ineq = conditional_inequality_loss(y_pde_pred, P_time_pde, threshold_p_scaled) #* 5 #weighting by factor of 5
    
    total_loss = loss_hotanddry + loss_fallow + loss_pde + lambda_ineq * loss_ineq #mulitple inequality loss by lambda to weight

    return total_loss, loss_hotanddry, loss_fallow, loss_pde, loss_ineq

class PINNEnsemble:
    def __init__(self, input_dim, n_models=10, device = None):
        self.n_models = n_models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.optimizers = []
        
        #create n_models with different random initializations
        for i in range(n_models):
            model = nn.Sequential(
                nn.Linear(input_dim, 60), nn.ReLU(),
                nn.Linear(60, 60), nn.ReLU(),
                nn.Linear(60, 1), nn.Sigmoid()
            ).to(device)
            
            self.models.append(model)
            self.optimizers.append(optim.AdamW(model.parameters()))
    
    def train_ensemble(self, x_hotanddry, y_hotanddry_target,
                   x_fallow_t, y_fallow_target,
                   x_pde, P_time_pde, threshold_p_scaled,
                   lambda_ineq,
                   epochs=2000, subsample_fraction=0.8): 
        """
        x_pde: full set of collocation points
        P_time_pde: P_time values for collocation points
        subsample_fraction: fraction of collocation points to use per model per epoch
        """
        
        n_collocation = len(x_pde)
        subsample_size = int(n_collocation * subsample_fraction)
                
        for epoch in range(epochs):
            epoch_losses = {f'model_{i}': {} for i in range(self.n_models)}
            
            for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                model.train()
                optimizer.zero_grad()
                
                #collocation subsampling
                subsample_idx = np.random.choice(
                    n_collocation, 
                    size=subsample_size, 
                    replace=False
                )
                x_pde_subset = x_pde[subsample_idx]
                P_time_subset = P_time_pde[subsample_idx]
                
                #forward pass with BC data
                y_hotanddry_pred = model(x_hotanddry)
                y_fallow_pred = model(x_fallow_t)
                
                #PDE residual
                pde_residual, y_pde_pred = self.compute_pde_residual(model, x_pde_subset)
                
                #loss calculation
                total_loss, loss_hotanddry, loss_fallow, loss_pde, loss_ineq = calculate_loss(
                    y_hotanddry_pred, y_fallow_pred, pde_residual,
                    y_hotanddry_target, y_fallow_target, y_pde_pred, lambda_ineq,
                    P_time_pde=P_time_subset, threshold_p_scaled = threshold_p_scaled  
                )

                total_loss.backward()
                optimizer.step()
                
                #store losses
                epoch_losses[f'model_{model_idx}'] = {
                    'total': total_loss.item(),
                    'pde': loss_pde.item(),
                    'hotanddry': loss_hotanddry.item(),
                    'fallow': loss_fallow.item(),
                    'ineq': loss_ineq.item()  
                }
            
            #print progress (ensemble average)
            if epoch % 100 == 0:
                avg_total = np.mean([epoch_losses[f'model_{i}']['total'] for i in range(self.n_models)])
                avg_pde = np.mean([epoch_losses[f'model_{i}']['pde'] for i in range(self.n_models)])
                avg_hotanddry = np.mean([epoch_losses[f'model_{i}']['hotanddry'] for i in range(self.n_models)])
                avg_fallow = np.mean([epoch_losses[f'model_{i}']['fallow'] for i in range(self.n_models)])
                avg_ineq = np.mean([epoch_losses[f'model_{i}']['ineq'] for i in range(self.n_models)])

                print(f'Epoch {epoch}: Ensemble Avg Loss = {avg_total:.6f}')
                print(f'    pde loss: {avg_pde:.6f} | '
                    f'hotanddry loss: {avg_hotanddry:.6f} | '
                    f'fallow loss: {avg_fallow:.6f} | '
                    f'ineq loss: {avg_ineq:.6f}')
                print()

    @staticmethod
    #compute pde residual
    def compute_pde_residual(model, x_pde):
        omega = 2 * np.pi / 24
        #first derivative
        y_pde = model(x_pde)
        dyt = torch.autograd.grad(
            outputs=y_pde,
            inputs=x_pde,
            grad_outputs=torch.ones_like(y_pde),
            create_graph=True,
            retain_graph=True
        )[0]
        sum_dyt = dyt.sum(dim=-1, keepdim=True)
        
        #second derivative
        laplacian = torch.autograd.grad(
            outputs=sum_dyt,
            inputs=x_pde,
            grad_outputs=torch.ones_like(sum_dyt),
            create_graph=True,
            retain_graph=True
        )[0].sum(dim=-1, keepdim=True)
        
        pde_residual = laplacian + (omega**2) * sum_dyt
        return pde_residual, y_pde
    
    def predict(self, x):
        #returns mean and std deviation of predictions
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
