import torch
from torch import save as torch_save
from torch import load as torch_load
from torch.nn import L1Loss as mean_absolute_error
from tqdm import tqdm
from numpy import mean, inf
from itertools import product

from matplotlib import pyplot as plt

from sklearn.metrics import r2_score


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
def validation(model, val_loader, criterion, device=torch.device('cuda')):
    y_test, y_pred = [], []

    for X, target in val_loader:
        X = X.to(device)
        output = model(X).cpu() # (-1), on cpu

        y_test.append(target)
        y_pred.append(output)

    y_test = torch.cat(y_test, dim=0) # (N), on cpu
    y_pred = torch.cat(y_pred, dim=0) # (N), on cpu
    
    loss = criterion(y_pred, y_test).item()
    # r2 = r2_score(y_test, y_pred)
    # r2 = 1 - (y_test-y_pred).pow(2).mean(axis=0) / y_test.var(axis=0, unbiased=False)
    # r2 = r2.mean().item()
    r2 = 0
    mae = mean_absolute_error()(y_pred, y_test).item()
            
    return loss, r2, mae


class Trainer:
    def __init__(self, criterion, device, save_path='checkpoints/model_best.pt'):
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.r2_results = []
        self.mae_results = []

        self.best_model = None
        self.best_loss = None
        self.best_epoch = None
    

    def train(self, model, optimizer, train_loader, val_loader, patience=10, epochs=100):
        early_stopping = EarlyStopping(save_path=self.save_path, patience=patience, delta=0)
        criterion = self.criterion
        device = self.device

        for epoch in range(epochs):
            train_loss = []
            progress_bar = tqdm(train_loader)
            progress_bar.set_description(f'Epoch {epoch:2d}')
            len_progress = len(progress_bar)

            for i, (X, target) in enumerate(progress_bar):
                model.train()
                X = X.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                progress_bar.set_postfix({'Train Loss': f'{loss.item():.5f}'})

                if i == len_progress-1:
                    model.eval()
                    train_loss = mean(train_loss)
                    self.train_losses.append(train_loss)
                    val_loss, val_r2, val_mae = validation(model, val_loader, criterion, device)

                    self.val_losses.append(val_loss)
                    self.r2_results.append(val_r2)
                    self.mae_results.append(val_mae)

                    progress_bar.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                        'Valid Loss': f'{val_loss:.4f}',
                        'Valid R2': f'{val_r2:.4f}',
                        'Valid MAE': f'{val_mae:.4f}'
                    })
            
            early_stopping(model, val_loss, epoch)
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
        test_loss, test_r2, test_mae = validation(self.best_model, test_loader, self.criterion, self.device)
        print(f'Test Loss: {test_loss:.4f} | Test R2: {test_r2:.4f} | Test MAE: {test_mae:.4f}')
        
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