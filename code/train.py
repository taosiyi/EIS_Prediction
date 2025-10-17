import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from dataprocess import train_tensor_build, create_dataloaders
import os

def plot_training_losses(train_losses, val_losses, model_dir, model_name):
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs(model_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'{model_name}_loss_plot.png'))
    plt.close()
    print(f"✅ {model_name} loss plot saved to {model_dir}")

def train_model(model, directory_path, device, model_dir, model_name,
                rest_len=14, eis_len=107,
                epochs=100, lr=1e-4, edge_index=None):

    input_tensor, output_tensor = train_tensor_build(directory_path, model_dir, rest_len, eis_len)
    print('rest(input_tensor.shape): ',input_tensor.shape)
    print('eis:(output_tensor.shape): ',output_tensor.shape)
    train_loader, val_loader = create_dataloaders(input_tensor, output_tensor)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                batch_edge_index = edge_index
            else:
                x, y, batch_edge_index = batch
            x = x.to(device)
            y = y.to(device)
            if batch_edge_index is not None:
                batch_edge_index = batch_edge_index.to(device)

            optimizer.zero_grad()

            if 'GCN' in model_name:
                pred = model(x, batch_edge_index)
            else:
                pred = model(x)

            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    batch_edge_index = edge_index
                else:
                    x, y, batch_edge_index = batch

                x = x.to(device)
                y = y.to(device)
                if batch_edge_index is not None:
                    batch_edge_index = batch_edge_index.to(device)

                if 'GCN' in model_name:
                    pred = model(x, batch_edge_index)
                else:
                    pred = model(x)
                
                total_val_loss += criterion(pred, y).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.to('cpu').state_dict(), os.path.join(model_dir, f'best_model.pth'))
            model.to(device)
    
    torch.save(model.to('cpu').state_dict(), os.path.join(model_dir, f'final_model.pth'))
    plot_training_losses(train_losses, val_losses, model_dir, model_name)



