import torch
from torch import save as torch_save
from torch import load as torch_load
from tqdm import tqdm
from numpy import mean, inf
from itertools import product

from matplotlib import pyplot as plt


def calc_profit(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = y_pred[:, -24:]
    y_true = y_true[:, -24:]
    profit = torch.zeros(y_true.size())
    util_condition = (y_true>=9.9)
    error = (y_true-y_pred).abs() / 99.0 * 100

    profit[util_condition & (error<=6)] = y_true[util_condition & (error <= 6)] * 4
    profit[util_condition & (error>6) & (error<=8)] = y_true[util_condition & (error>6) & (error<=8)] * 3
    profit = profit.sum(dim=1).mean().item()

    return profit


class EarlyStopping:
    def __init__(self, save_path, patience=10, delta=0):
        self.patience = patience
        self.best_loss = inf
        self.early_stop = False
        self.delta = delta
        self.counter = 0
        self.best_model_save_path = save_path


    def __call__(self, model, val_loss, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch_save(model.state_dict(), self.best_model_save_path)

        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        

@torch.no_grad()
def validation(model, val_loader, device=torch.device('cuda')):
    y_test, y_pred = [], []
    relu = torch.relu

    for x1, x2, target in val_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        output = relu(model(x1, x2).cpu()) # (-1 x P), on cpu

        y_test.append(target)
        y_pred.append(output)

    y_test = torch.cat(y_test, dim=0) # (N x P), on cpu
    y_pred = torch.cat(y_pred, dim=0) # (N x P), on cpu
    
    loss = (y_pred-y_test).pow(2).mean().item()
    mae = (y_pred-y_test).abs().mean().item()

    r2 = (y_pred-y_test).pow(2).mean(dim=1)  # (N)
    r2 = 1 - r2 / y_test.var(dim=1, unbiased=False)  # (N)
    r2 = r2.mean().item()

    profit = calc_profit(y_pred, y_test)

    return loss, r2, mae, profit


class Trainer:
    def __init__(self, criterion, device, save_path='checkpoints/model_best.pt'):
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.r2_results = []
        self.mae_results = []
        self.profit_results = []

        self.best_model = None
        self.best_loss = None
        self.best_epoch = None
    

    def train(self, model, optimizer, train_loader, val_loader, patience=10, epochs=100):
        early_stopping = EarlyStopping(save_path=self.save_path, patience=patience, delta=0)
        criterion = self.criterion
        device = self.device

        for epoch in range(epochs):
            train_loss = []
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch:3d}')
            pbar_update = pbar.update
            set_postfix = pbar.set_postfix

            model.train()
            for x1, x2, target in train_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(x1, x2)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                pbar_update()
                set_postfix({'Train Loss': f'{loss.item():.4f}'})
            
            model.eval()
            train_loss = mean(train_loss)
            self.train_losses.append(train_loss)
            val_loss, val_r2, val_mae, val_profit = validation(model, val_loader, device)

            self.val_losses.append(val_loss)
            self.r2_results.append(val_r2)
            self.mae_results.append(val_mae)
            self.profit_results.append(val_profit)

            set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Valid Loss': f'{val_loss:.4f}',
                'Valid R2': f'{val_r2:.4f}',
                'Valid MAE': f'{val_mae:.4f}',
                'Valid Profit': f'{val_profit:.1f}'
            })
            pbar.close()
            
            early_stopping(model, -val_profit, epoch)
            if early_stopping.early_stop:
                print('Early Stopped')
                break
        
        model.load_state_dict(torch_load(early_stopping.best_model_save_path))
        self.best_model = model
        self.best_loss = early_stopping.best_loss
        self.best_epoch = early_stopping.best_epoch
        
        return self.best_model


    @torch.no_grad()
    def test(self, test_loader):
        self.best_model.eval()
        test_loss, test_r2, test_mae, test_profit = validation(self.best_model, test_loader, self.device)
        print(f'Test Loss: {test_loss:.4f} | Test R2: {test_r2:.4f} | Test MAE: {test_mae:.4f} | Test Profit: {test_profit:.1f}')
        
        return None


    def plot_losses(self, plot_title='fuck', save_filename=None):
        fig, sub = plt.subplots(1, 1, dpi=100, figsize=(7, 5))
        epochs = len(self.train_losses)
        sub.plot(range(epochs), self.train_losses, color='r', label='train loss')
        sub.plot(range(epochs), self.val_losses, color='b', label='valid loss')
        sub.axvline(x=self.best_epoch, color='k', linestyle='--', label='best epoch')

        sub.set_ylim(0, 4.0)
        sub.set_xlabel('epoch')
        sub.set_ylabel('loss')
        sub.set_title(plot_title)
        sub.legend()

        if save_filename:
            fig.savefig(save_filename)
            print(f'Saving Process Complete. Directory: {save_filename}')

        return None


class GridSearch(Trainer):
    def __init__(self, criterion, device, temp_save_path='checkpoints/model_temp.pt'):
        self.criterion = criterion
        self.device = device
        self.temp_save_path = temp_save_path
        self.best_loss = inf
        

    def train(self):
        pass


    def train_by_grid(self, Model, basic_params, param_grid, optimizer_function, train_loader, val_loader, lr=5e-5, patience=3, epochs=20, save_filename='checkpoints/model_best.pt'):
        param_grid_keys = list(param_grid.keys())
        all_cases = product(*param_grid.values())
        print(f'Total {len(list(all_cases)):3d} cases are going to be searched.')
        del all_cases

        for i, param_values in enumerate(product(*param_grid.values())):
            params = dict(zip(param_grid_keys, param_values))
            train_model = Trainer(self.criterion, self.device, self.temp_save_path, self.masked)
            model = Model(**basic_params, **params).to(self.device)
            optimizer = optimizer_function(params=model.parameters(), lr=lr)
            
            print(f'Setting {i:3d} of Parameters Grid is now on progres.')
            best_model_param = train_model.train(model, optimizer, train_loader, val_loader, patience=patience, epochs=epochs)

            if train_model.best_loss < self.best_loss:
                self.best_model = best_model_param
                self.best_loss = train_model.best_loss
                self.best_epoch = train_model.best_epoch

                self.train_losses = train_model.train_losses
                self.val_losses = train_model.val_losses
                self.r2_results = train_model.r2_results
                self.mae_results = train_model.mae_results
                self.best_params_ = param_values # temporary
        
                if save_filename:
                    torch.save(self.best_model.state_dict(), save_filename)
        
        self.best_params_ = dict(zip(param_grid_keys, self.best_params_))
        print(self.best_params_)

        return self.best_model
    

    def test(self, test_loader):
        super(GridSearch, self).test(test_loader)

    
    def plot_losses(self, plot_title='fuck', save_filename=None):
        super(GridSearch, self).plot_losses(plot_title, save_filename)