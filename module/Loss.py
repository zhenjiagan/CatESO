import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMSELoss(nn.Module):
    def __init__(self, coefficient=1.0):
        super(RelativeMSELoss, self).__init__()
        self.coefficient = coefficient
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)

        N = y_pred.shape[0]

        loss_abs = self.mse_loss(y_pred, y_true)

        if N < 2:
            return loss_abs

        pred_diff = y_pred - y_pred.t()
        true_diff = y_true - y_true.t()

        rel_errors_matrix = (pred_diff - true_diff) ** 2

        upper_triangle = torch.triu(rel_errors_matrix, diagonal=1)

        num_pairs = N * (N - 1) / 2.0
        
        loss_rel = upper_triangle.sum() / num_pairs

        total_loss = loss_abs + self.coefficient * loss_rel

        return total_loss