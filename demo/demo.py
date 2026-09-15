import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
import numpy as np
import random
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score # For evaluation metrics
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.rome import MoE, FairMoE_AS, FairMoE_S, MLP, BaselineMLP, train_rome_moe_with_dro, train_moe_model, evaluate_moe_model, train_baseline_model, evaluate_baseline_model, is_numeric_continuous

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

SEED = 123
set_all_seeds(SEED)

def seed_worker(worker_id): # for data loaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

TRAIN_CSV_FILE_PATH = "data/train.csv"
TEST_CSV_FILE_PATH = "data/test.csv"

Y_COL_NAME = "Y"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)


def load_and_prepare_data(train_csv_path, test_csv_path, y_col, random_state=42):

    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
    except FileNotFoundError as e:
        print(f"Error: A data file was not found. Please check the path. Details: {e}")
        raise

    a_cols = sorted([col for col in train_df.columns if col.startswith('A') and col[1:].isdigit()])
    s_cols = sorted([col for col in train_df.columns if col.startswith('S') and col[1:].isdigit()])

    print(f"\nAuto-detected {len(a_cols)} 'A' feature columns: {a_cols}")
    print(f"Auto-detected {len(s_cols)} 'S' feature columns: {s_cols}")

    Y_train_np = train_df[y_col].values.astype(np.float32)
    A_train_np = train_df[a_cols].values.astype(np.float32)
    S_train_np = train_df[s_cols].values.astype(np.float32)

    Y_test_np = test_df[y_col].values.astype(np.float32)
    A_test_np = test_df[a_cols].values.astype(np.float32)
    S_test_np = test_df[s_cols].values.astype(np.float32)

    A_train_tensor = torch.from_numpy(A_train_np).float()
    S_train_tensor = torch.from_numpy(S_train_np).float()
    Y_train_tensor = torch.from_numpy(Y_train_np).float().unsqueeze(1)

    A_test_tensor = torch.from_numpy(A_test_np).float()
    S_test_tensor = torch.from_numpy(S_test_np).float()
    Y_test_tensor = torch.from_numpy(Y_test_np).float().unsqueeze(1)

    return (A_train_tensor, S_train_tensor, Y_train_tensor, 
            A_test_tensor, S_test_tensor, Y_test_tensor,
            train_df, test_df)

def derive_groups_from_S(S_train_tensor, S_test_tensor, num_groups, random_state=42):
    """
    Derive latent groups using KMeans clustering on S features.
    Accepts tensors and converts them to numpy for KMeans.
    """
    # Convert tensors to numpy for sklearn
    S_train_np = S_train_tensor.numpy() if torch.is_tensor(S_train_tensor) else S_train_tensor
    S_test_np = S_test_tensor.numpy() if torch.is_tensor(S_test_tensor) else S_test_tensor
    
    print(f"Deriving {num_groups} latent groups from S features using KMeans clustering...")
    kmeans = KMeans(n_clusters=num_groups, random_state=random_state, n_init='auto')
    kmeans.fit(S_train_np)
    
    G0_test = kmeans.predict(S_test_np).astype(np.int64)
    print(f"KMeans derived {num_groups} groups - unique values (test): {np.unique(G0_test)}")
    
    return G0_test, kmeans

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    NUM_DERIVED_GROUPS_MoE_full = 2
    NUM_DERIVED_GROUPS_MoE_fair_1 = 3
    NUM_DERIVED_GROUPS_MoE_fair_2 = 2

    (A_train, S_train, Y_train,
    A_test, S_test, Y_test,
    train_df, test_df) = load_and_prepare_data(TRAIN_CSV_FILE_PATH, TEST_CSV_FILE_PATH, Y_COL_NAME)

    X_combined_train = torch.cat([A_train, S_train], dim=-1)
    X_combined_test = torch.cat([A_test, S_test], dim=-1)

    baseline_model_full = BaselineMLP(input_size=X_combined_train.shape[1],
                                     output_size=1, hidden_size=128).to(device)
    baseline_model_fair = BaselineMLP(input_size=A_train.shape[1],
                                      output_size=1, hidden_size=64).to(device)
    
    moe_model_full = MoE(input_size_A=A_train.shape[1], input_size_S=S_train.shape[1],
                        output_size=1, num_experts=3, expert_hidden_size=64, gating_hidden_size=32).to(device)
    moe_model_fair_1 = FairMoE_S(input_size_A=A_train.shape[1], input_size_S=S_train.shape[1],
                        output_size=1, num_experts=4, expert_hidden_size=64, gating_hidden_size=16).to(device)
    moe_model_fair_2 = FairMoE_AS(input_size_A=A_train.shape[1], input_size_S=S_train.shape[1],
                        output_size=1, num_experts=4, expert_hidden_size=32, gating_hidden_size=16).to(device)

    moe_optim_full = Adam(moe_model_full.parameters(), lr=0.01)
    baseline_optim = Adam(baseline_model_full.parameters(), lr=0.01)
    baseline_optim_fair = Adam(baseline_model_fair.parameters(), lr=0.01)
    moe_optim_fair_1 = Adam(moe_model_fair_1.parameters(), lr=0.01) #FairMoE-S
    moe_optim_fair_2 = Adam(moe_model_fair_2.parameters(), lr=0.001) #FairMoE-AS
    loss_fn = nn.MSELoss()

    g = set_all_seeds(SEED)
    num_epochs = 30
    batch_size = 64
    moe_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(A_train, S_train, Y_train), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    baseline_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_combined_train, Y_train), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    baseline_fair_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(A_train, Y_train), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    print(f"\n--- Training Standard MoE Model for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        total_loss = sum(train_moe_model(b_A, b_S, b_Y, moe_model_full, loss_fn, moe_optim_full, device)
                         for b_A, b_S, b_Y in moe_train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"MoE - Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {total_loss / len(moe_train_loader):.4f}")

    print(f"\n--- Training Baseline MLP Model for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        total_loss = sum(train_baseline_model(b_X, b_Y, baseline_model_full, loss_fn, baseline_optim, device)
                         for b_X, b_Y in baseline_train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Baseline - Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {total_loss / len(baseline_train_loader):.4f}")

    print(f"\n--- Training Fair Baseline MLP Model for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        total_loss = sum(train_baseline_model(b_A, b_Y, baseline_model_fair, loss_fn, baseline_optim_fair, device)
                         for b_A, b_Y in baseline_fair_train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Fair Baseline - Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {total_loss / len(baseline_fair_train_loader):.4f}")
    
    ALPHA_FAIRMOE_S = 0.05
    ALPHA_FAIRMOE_AS = 0.05

    print(f"\n--- Training Fair MoE Model for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        total_loss = sum(train_rome_moe_with_dro(b_A, b_S, b_Y, moe_model_fair_1, moe_optim_fair_1, device, ALPHA_FAIRMOE_S)
                         for b_A, b_S, b_Y in moe_train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Fair MoE - Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {total_loss / len(moe_train_loader):.4f}")
    
    print(f"\n--- Training Fair MoE Model for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        total_loss = sum(train_rome_moe_with_dro(b_A, b_S, b_Y, moe_model_fair_2, moe_optim_fair_2, device, ALPHA_FAIRMOE_AS)
                         for b_A, b_S, b_Y in moe_train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Fair MoE - Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {total_loss / len(moe_train_loader):.4f}")

    print("\n--- Final Overall Evaluation on TEST DATA ---")
    
    final_baseline_loss = evaluate_baseline_model(X_combined_test, Y_test, baseline_model_full, loss_fn, device)
    final_baseline_fair_loss = evaluate_baseline_model(A_test, Y_test, baseline_model_fair, loss_fn, device)
    final_moe_loss = evaluate_moe_model(A_test, S_test, Y_test, moe_model_full, loss_fn, device)
    final_moe_fair_loss_1 = evaluate_moe_model(A_test, S_test, Y_test, moe_model_fair_1, loss_fn, device)
    final_moe_fair_loss_2 = evaluate_moe_model(A_test, S_test, Y_test, moe_model_fair_2, loss_fn, device)

    print(f"Baseline MLP Final Test MSE: {final_baseline_loss:.4f}")
    print(f"Fair Baseline MLP Final Test MSE: {final_baseline_fair_loss:.4f}")
    print(f"Standard MoE Final Test MSE: {final_moe_loss:.4f}")
    print(f"Fair MoE 1 Final Test MSE: {final_moe_fair_loss_1:.4f}")
    print(f"Fair MoE 2 Final Test MSE: {final_moe_fair_loss_2:.4f}")

    print(f"\n--- Evaluating Models on Subgroups Defined by: {', '.join(EVALUATION_COLS)} ---")

    # --- Subgroup Evaluation ---
    quantile_groups_df = pd.DataFrame(index=test_df.index)

    for col in EVALUATION_COLS:
        print(f"\nProcessing column: {col}")

        if is_numeric_continuous(train_df[col]):
            print(f"  -> Treating {col} as continuous variable")
            
            if SPLIT_METHOD == "quartile":
                q25, q50, q75 = train_df[col].quantile([0.25, 0.5, 0.75])
                bins = [-np.inf, q25, q50, q75, np.inf]
                labels = [f"{col}_Q1", f"{col}_Q2", f"{col}_Q3", f"{col}_Q4"]
                print(f"  -> Quartile splits: Q1≤{q25:.3f}, Q2≤{q50:.3f}, Q3≤{q75:.3f}, Q4>{q75:.3f}")
            elif SPLIT_METHOD == "median":
                median_val = train_df[col].median()
                bins = [-np.inf, median_val, np.inf]
                labels = [f"{col}_Low", f"{col}_High"]
                print(f"  -> Median split at: {median_val:.3f}")
            
            quantile_groups_df[f'{col}_quantile'] = pd.cut(test_df[col], bins=bins, labels=labels, include_lowest=True)
            
        else:
            print(f"  -> Treating {col} as categorical variable")
            
            unique_values = sorted(train_df[col].dropna().unique())
            print(f"  -> Unique values: {unique_values}")
            
            def create_categorical_labels(x):
                if pd.isna(x):
                    return f"{col}_Missing"
                else:
                    return f"{col}_{x}"
            
            quantile_groups_df[f'{col}_quantile'] = test_df[col].apply(create_categorical_labels)

    test_composite_groups = quantile_groups_df.apply(lambda row: '_'.join(row.astype(str)), axis=1).values

    unique_eval_groups = np.sort(np.unique(test_composite_groups))
    print(f"Created {len(unique_eval_groups)} intersectional subgroups for evaluation.")

    Y_test_np_flat = Y_test.squeeze().numpy()
    results = []

    y_hat_moe_standard = moe_model_full(A_test.to(device), S_test.to(device)).cpu().detach().numpy().flatten()
    y_hat_baseline = baseline_model_full(X_combined_test.to(device)).cpu().detach().numpy().flatten()
    y_hat_baseline_fair = baseline_model_fair(A_test.to(device)).cpu().detach().numpy().flatten()
    y_hat_moe_fair_1 = moe_model_fair_1(A_test.to(device), S_test.to(device)).cpu().detach().numpy().flatten()
    y_hat_moe_fair_2 = moe_model_fair_2(A_test.to(device), S_test.to(device)).cpu().detach().numpy().flatten()
    
    for idx_g, group_val in enumerate(unique_eval_groups):
        indices = (test_composite_groups == group_val)
        # Skip if group is empty or too small to calculate R-squared reliably
        if not np.any(indices) or len(Y_test_np_flat[indices]) < 2:  
            continue

        Y_g = Y_test_np_flat[indices]
        
        group_metrics = {'Group': group_val, 'GroupSize': len(Y_g)}

        pred_base_g = y_hat_baseline[indices]
        group_metrics['MSE_BaselineMLP'] = mean_squared_error(Y_g, pred_base_g)

        pred_base_fair_g = y_hat_baseline_fair[indices]
        group_metrics['MSE_FairBaselineMLP'] = mean_squared_error(Y_g, pred_base_fair_g)

        pred_moe_g = y_hat_moe_standard[indices]
        group_metrics['MSE_StandardMoE'] = mean_squared_error(Y_g, pred_moe_g)

        pred_moe_fair_g_1 = y_hat_moe_fair_1[indices]
        group_metrics['MSE_FairMoE_1'] = mean_squared_error(Y_g, pred_moe_fair_g_1)

        pred_moe_fair_g_2 = y_hat_moe_fair_2[indices]
        group_metrics['MSE_FairMoE_2'] = mean_squared_error(Y_g, pred_moe_fair_g_2)

        results.append(group_metrics)

    results_df = pd.DataFrame(results).set_index('Group')
    mse_cols = [col for col in results_df.columns if 'MSE' in col]
    r2_cols = [col for col in results_df.columns if 'R2' in col]

    mse_df = results_df[mse_cols].rename(columns=lambda c: c.replace('MSE_', ''))
    r2_df = results_df[r2_cols].rename(columns=lambda c: c.replace('R2_', ''))

    print("\n--- Per-Group MSE on TEST DATA ---")
    print(mse_df)

    print(f"\n--- Max MSE Across All Subgroups (Worst-Group Performance) ---")
    print(mse_df.max(axis=0))

    # EVALUATION ON ENTIRE TEST SET:
    print("\n" + "="*80)
    print("--- FINAL COMPREHENSIVE EVALUATION ON ENTIRE TEST DATASET ---")
    print("="*80)

    Y_test_full = Y_test_np_flat
    
    final_results = {}

    
    # Baseline MLP (Full)
    mse_baseline = mean_squared_error(Y_test_full, y_hat_baseline)
    final_results['BaselineMLP_Full'] = {'MSE': mse_baseline}
    
    # Fair Baseline MLP
    mse_baseline_fair = mean_squared_error(Y_test_full, y_hat_baseline_fair)
    final_results['BaselineMLP_Fair'] = {'MSE': mse_baseline_fair}
    
    # Standard MoE
    mse_moe = mean_squared_error(Y_test_full, y_hat_moe_standard)
    final_results['StandardMoE'] = {'MSE': mse_moe}
    
    # Fair MoE (if uncommented)
    mse_moe_fair_1 = mean_squared_error(Y_test_full, y_hat_moe_fair_1)
    final_results['FairMoE_1'] = {'MSE': mse_moe_fair_1}

    # Fair MoE (if uncommented)
    mse_moe_fair_2 = mean_squared_error(Y_test_full, y_hat_moe_fair_2)
    final_results['FairMoE_2'] = {'MSE': mse_moe_fair_2}

    final_summary_df = pd.DataFrame(final_results).T
    final_summary_df = final_summary_df.round(6)
    
    print("\n--- FINAL TEST SET PERFORMANCE SUMMARY ---")
    print("-" * 60)
    print(final_summary_df)

if __name__ == "__main__":
    from itertools import product
    
    EVALUATION_COLS_OPTIONS = [
        ["S1", "S2", "S3"]]

    SPLIT_METHOD_OPTIONS = ["median"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output/demo_python_seed_{SEED}_{timestamp}.txt"
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # Ensure immediate write to file
        
        def flush(self):
            pass

    sys.stdout = Logger(output_filename)
    
    print(f"Starting multi-configuration analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {output_filename}")
    print(f"Total configurations to run: {len(EVALUATION_COLS_OPTIONS) * len(SPLIT_METHOD_OPTIONS)}")
    print("=" * 80 + "\n")
    
    for config_num, (eval_cols, split_method) in enumerate(product(EVALUATION_COLS_OPTIONS, SPLIT_METHOD_OPTIONS), 1):
        
        EVALUATION_COLS = eval_cols
        SPLIT_METHOD = split_method
        
        eval_cols_str = "_".join(eval_cols)
        print("\n" + "=" * 80)
        print(f"CONFIGURATION {config_num}")
        print(f"Evaluation Columns: {eval_cols}")
        print(f"Split Method: {split_method}")
        print(f"Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        try:
            main()
            
            print(f"\nConfiguration {config_num} completed successfully")
        except Exception as e:
            print(f"\nERROR in Configuration {config_num}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
    
    # Print overall footer
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"All configurations completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total configurations run: {len(EVALUATION_COLS_OPTIONS) * len(SPLIT_METHOD_OPTIONS)}")
    print(f"Results saved to: {output_filename}")