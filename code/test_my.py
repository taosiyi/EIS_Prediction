import os
import re
import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, max_error
import torch

def test_result_get_single(rest_len, model_path, cell_path, input_file, model, edge_index=None):

    scaler_path = os.path.join(model_path, "scaler_input.pth")
    scaler = joblib.load(scaler_path)
    input_tensor = None
    file_path = os.path.join(cell_path, input_file)
    df = pd.read_pickle(file_path)
    segment = df[['Ecell/V', '<I>/mA', 'Temperature/°C', 'time']].iloc[:rest_len].values

    input_scaled = scaler.transform(segment)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pth")))
    model.to(device)
    model.eval()

    # edge_index
    if edge_index is None:
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 3, 0, 1, 1, 2]
        ], dtype=torch.long).to(device)
    else:
        edge_index = edge_index.to(device)
        
    with torch.no_grad():
        x = input_tensor.to(device)
        if 'GCN' in model.__class__.__name__ or hasattr(model, 'gcn1'):
            output = model(x, edge_index)
        else:
            output = model(x)
    
    scaler_output = joblib.load(os.path.join(model_path, "scaler_output.pth"))
    output = output.cpu().numpy()
    output = scaler_output.inverse_transform(output.reshape(-1, 2)).reshape(output.shape)
    prediction = output
    prediction = output.squeeze(0)
    return prediction

def save_single_eis_results_as_pkl(eis_len, prediction, result_file, cell_path, output_file):

    eis_path = os.path.join(cell_path, output_file)
    df = pd.read_pickle(eis_path)
    freq_meas = df['freq/Hz'].values
    meas_real = df['|Z|/Ohm'].values
    meas_imag = df['Phase(Z)/deg'].values

    result_df = pd.DataFrame({
        'freq': freq_meas[1:eis_len],
        'pred_mag': prediction[:, 0],
        'pred_ph': prediction[:, 1],
        'meas_mag': meas_real[1:eis_len],
        'meas_ph': meas_imag[1:eis_len]
    })

    directory = os.path.dirname(result_file)
    os.makedirs(directory, exist_ok=True)

    with open(result_file, 'wb') as f:
        pickle.dump(result_df, f)

def extract_number_eis(file_name):
    match = re.search(r'eis_(\d+)_', file_name)
    return int(match.group(1)) if match else -1  # Return -1 if no match found

def extract_number_rest(file_name):
    match = re.search(r'rest_(\d+)_', file_name)
    return int(match.group(1)) if match else -1  # Return -1 if no match found

def extract_info(file_name):
    eis_match = re.search(r'eis_(\d+)_', file_name)
    cap_match = re.search(r'cap_(\d+\.?\d*)\.pkl', file_name)
    
    eis_number = int(eis_match.group(1)) if eis_match else None
    cap_value = float(cap_match.group(1)) if cap_match else None
    
    return eis_number, cap_value

def Z_re_im_get(data):
    meas_mag = data['meas_mag'] * 1000  #mΩ
    meas_ph = data['meas_ph']
    pred_mag = data['pred_mag'] * 1000  #mΩ
    pred_ph = data['pred_ph']
    frequency = data['freq']  

    theta_meas_rad = np.deg2rad(data['meas_ph'])  
    theta_pred_rad = np.deg2rad(data['pred_ph'])

    meas_real = data['meas_mag'] * np.cos(theta_meas_rad) * 1000   # mΩ
    meas_imag = -data['meas_mag'] * np.sin(theta_meas_rad) * 1000
    pred_real = data['pred_mag'] * np.cos(theta_pred_rad) * 1000
    pred_imag = -data['pred_mag'] * np.sin(theta_pred_rad) * 1000
    return frequency,meas_mag,meas_ph,pred_mag,pred_ph,meas_real,meas_imag,pred_real,pred_imag

def calculate_metrics(data):

    frequency,meas_mag,meas_ph,pred_mag,pred_ph,meas_real,meas_imag,pred_real,pred_imag = Z_re_im_get(data)
    
    r2_real = r2_score(meas_real, pred_real)
    r2_imag = r2_score(meas_imag, pred_imag)
    r2_magnitude = r2_score(meas_mag, pred_mag)
    r2_phase = r2_score(meas_ph, pred_ph)
    
    rmse_real = np.sqrt(mean_squared_error(meas_real, pred_real))
    rmse_imag = np.sqrt(mean_squared_error(meas_imag, pred_imag))
    rmse_magnitude = np.sqrt(mean_squared_error(meas_mag, pred_mag))
    rmse_phase = np.sqrt(mean_squared_error(meas_ph, pred_ph))
    
    mape_real = np.mean(np.abs((meas_real - pred_real) / np.clip(meas_real, 1e-10, None))) * 100
    mape_imag = np.mean(np.abs((meas_imag - pred_imag) / np.clip(meas_imag, 1e-10, None))) * 100
    mape_magnitude = np.mean(np.abs((meas_mag - pred_mag) / np.clip(meas_mag, 1e-10, None))) * 100
    mape_phase = np.mean(np.abs((meas_ph - pred_ph) / np.clip(meas_ph, 1e-10, None))) * 100
    
    max_real = max_error(meas_real, pred_real)
    max_imag = max_error(meas_imag, pred_imag)
    max_magnitude = max_error(meas_mag, pred_mag)
    max_phase = max_error(meas_ph, pred_ph)
    
    return {
        'r2_real': r2_real, 'r2_imag': r2_imag, 'r2_magnitude': r2_magnitude, 'r2_phase': r2_phase,
        'rmse_real': rmse_real, 'rmse_imag': rmse_imag, 'rmse_magnitude': rmse_magnitude, 'rmse_phase': rmse_phase,
        'mape_real': mape_real, 'mape_imag': mape_imag, 'mape_magnitude': mape_magnitude, 'mape_phase': mape_phase,
        'max_real': max_real, 'max_imag': max_imag, 'max_magnitude': max_magnitude, 'max_phase': max_phase
    }

def error_excel_get(cell_path, save_path, cell):
    result = []
    predict_eis_files = [f for f in os.listdir(cell_path) if f.endswith(".pkl")]
    sorted_predict_eis_files = sorted(predict_eis_files, key=lambda x: extract_number_eis(x))

    for i, seg in enumerate(sorted_predict_eis_files):
        file_path = os.path.join(cell_path, seg)
        eis_number, cap_value = extract_info(file_path)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        metrics = calculate_metrics(data)
        result.append({
            'RPT': eis_number,
            'capacity': cap_value,
            **metrics
        })
    
    df_result = pd.DataFrame(result)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_result.to_excel(save_path, index=False)



