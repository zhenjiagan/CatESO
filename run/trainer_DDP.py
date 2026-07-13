import torch
import torch.nn as nn
import copy
import os
import numpy as np
from utils import set_seed, graph_collate_func, mkdir
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from prettytable import PrettyTable
from tqdm import tqdm
from torch import distributed
from module.Loss import RelativeMSELoss

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


class Trainer_Reg_DDP(object):
    def __init__(self, seed, model, optim, device, train_dataloader, val_dataloader, test_dataloader, val_sampler, test_sampler, output, alpha=1, scheduler=None, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.scheduler = scheduler  

        self.seed = seed

        self.best_model = None
        self.best_epoch = None
        self.best_rmse = 10
        self.best_r2 = -1
        self.best_test_metrics = {}

        # self.loss_func = nn.MSELoss(reduction='mean')
        self.loss_func = RelativeMSELoss(coefficient=1.0)

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_rmse_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output

        valid_metric_header = ["# Epoch", "PCC", "RMSE", "MAE", "R2", "Val_loss", "LR"]
        test_metric_header = ["# Epoch", "PCC", "RMSE", "MAE", "R2", "Test_loss", "LR"]
        best_model_metric_header = ["Best_Epoch", "PCC", "RMSE", "MAE", "R2", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss", "LR"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.best_model_table = PrettyTable(best_model_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        set_seed(self.seed)
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1

            current_lr = self.optim.param_groups[0]['lr']

            self.train_dataloader.sampler.set_epoch(self.current_epoch)

            train_loss = self.train_epoch()

            if distributed.get_rank() == 0:
                print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(train_loss) + " LR " + f"{current_lr:.6f}")
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss])) + [f"{current_lr:.6f}"]
                self.train_table.add_row(train_lst)
                self.train_loss_epoch.append(train_loss)
            distributed.barrier()

            val_loss, mae, rmse, pcc, r2, _, _ = self.test(dataloader="val")

            if distributed.get_rank() == 0:
                val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [pcc, rmse, mae, r2, val_loss])) + [f"{current_lr:.6f}"]
                self.val_table.add_row(val_lst)
                self.val_loss_epoch.append(val_loss)
                self.val_rmse_epoch.append(rmse)
                print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " RMSE "
                  + str(rmse) + " R2 " + str(r2) + " LR " + f"{current_lr:.6f}")
            distributed.barrier()

            if rmse <= self.best_rmse and r2 >= self.best_r2:
                self.best_model = copy.deepcopy(self.model)
                self.best_rmse = rmse
                self.best_r2 = r2
                self.best_epoch = self.current_epoch
                if distributed.get_rank() == 0:
                    torch.save(self.best_model.module.state_dict(), os.path.join(self.output_dir, f"best_model_epoch.pth"))

                distributed.barrier()
                best_test_loss, best_test_mae, best_test_rmse, best_test_pcc, best_test_r2, best_y_pred, best_y_label = self.test(dataloader="test", use_best_model=True)
                if distributed.get_rank() == 0:
                    best_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [best_test_pcc, best_test_rmse, best_test_mae, best_test_r2, best_test_loss]))
                    self.best_model_table.add_row(best_lst)
                    print(f'*** Best Model Updated at Epoch {self.best_epoch} ***')
                    print(f'Best Model Test: loss={best_test_loss:.4f}, RMSE={best_test_rmse:.4f}, PCC={best_test_pcc:.4f}, MAE={best_test_mae:.4f}, R2={best_test_r2:.4f}')
                self.best_test_metrics = {
                    "pcc": best_test_pcc,
                    "rmse": best_test_rmse,
                    "mae": best_test_mae,
                    "r2": best_test_r2,
                    "test_loss": best_test_loss,
                    "best_epoch": self.best_epoch,
                    "y_pred": best_y_pred,
                    "y_label": best_y_label,
                }
                distributed.barrier()

            if self.scheduler is not None:
                self.scheduler.step()      
                
        self.test_metrics = {
            "pcc": self.best_test_metrics.get("pcc"),
            "rmse": self.best_test_metrics.get("rmse"),
            "mae": self.best_test_metrics.get("mae"),
            "r2": self.best_test_metrics.get("r2"),
            "test_loss": self.best_test_metrics.get("test_loss"),
            "best_epoch": self.best_epoch,
        }
        y_pred = self.best_test_metrics.get("y_pred", [])
        y_label = self.best_test_metrics.get("y_label", [])

        if distributed.get_rank() == 0:
            print(f"Best model at epoch {self.best_epoch}: RMSE={self.test_metrics['rmse']:.4f}, PCC={self.test_metrics['pcc']:.4f}, MAE={self.test_metrics['mae']:.4f}, R2={self.test_metrics['r2']:.4f}")
            self.save_result()
        distributed.barrier()

        return self.test_metrics, y_pred, y_label

    def save_result(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        torch.save(self.best_model.module.state_dict(),
                   os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.module.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
       
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        best_model_prettytable_file = os.path.join(self.output_dir, "best_model_test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(best_model_prettytable_file, 'w') as fp:
            fp.write(self.best_model_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score= self.model(v_d, v_p, v_d_mask, v_p_mask)

            loss = self.loss_func(score.squeeze(-1), labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            
        loss_epoch = loss_epoch / num_batches
        return loss_epoch

    def test(self, dataloader="test", use_best_model=False):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            if use_best_model and self.best_model is not None:
                self.best_model.eval()
                model_to_use = self.best_model
            else:
                self.model.eval()
                model_to_use = self.model
                
            for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
                v_d, v_p, f, score = model_to_use(v_d, v_p, v_d_mask, v_p_mask)
                loss = self.loss_func(score.squeeze(-1), labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + score.to("cpu").tolist()
                
        y_label = torch.tensor(y_label, dtype=torch.float32, device=self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.float32, device=self.device)
        if dataloader == "test":
            y_pred = distributed_concat(y_pred, len(self.test_sampler.dataset))
            y_label = distributed_concat(y_label, len(self.test_sampler.dataset))
        else:
            y_pred = distributed_concat(y_pred, len(self.val_sampler.dataset))
            y_label = distributed_concat(y_label, len(self.val_sampler.dataset))

        y_pred = y_pred.to("cpu").tolist()
        y_label = y_label.to("cpu").tolist()
        
        test_loss = test_loss / num_batches
        pcc, _ = pearsonr(np.array(y_label).flatten(), np.array(y_pred).flatten())
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)
        return test_loss, mae, rmse, pcc, r2, y_pred, y_label
