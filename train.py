import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


from utils import *

def train_ensemble_with_early_stopping(train_embeddings, train_labels,
                                       validation_embeddings, validation_labels,
                                       MLP, N=10, batch_size=2024, epochs=10,
                                       model_patience=3, ensemble_patience=2):
    """
    Trains an ensemble of MLPs with individual early stopping and ensemble-level early stopping
    based on Negative Log Likelihood (NLL), Expected Calibration Error (ECE), and F1 Score.

    Parameters:
        train_embeddings (np.array): Training data embeddings.
        train_labels (np.array): Training labels.
        validation_embeddings (np.array): Validation data embeddings.
        validation_labels (np.array): Validation labels.
        N (int): Maximum number of models in the ensemble.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs per model.
        hidden_size (int): Number of hidden units for the first hidden layer (second layer uses a fixed 64 units).
        device (torch.device): Device to use for training.
        model_patience (int): Number of epochs without improvement for individual model early stopping.
        ensemble_patience (int): Number of consecutive ensemble additions with no improvement to trigger ensemble-level early stopping.

    Returns:
        ensemble_models (list): List of trained models in the ensemble.
        metrics (dict): Dictionary with keys 'ensemble_sizes', 'accuracies', 'nlls', 'eces', 'f1s' tracking metric evolution.
    """

    global device
    
    # Convert input data into PyTorch tensors
    train_X = torch.from_numpy(train_embeddings).float()
    train_y = torch.from_numpy(train_labels).long()
    val_X = torch.from_numpy(validation_embeddings).float()
    val_y = torch.from_numpy(validation_labels).long()

    # Create the training DataLoader
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    ensemble_models = []
    
    # Initialize metrics storage
    individual_accuracies = []
    individual_nlls   = []
    individual_eces   = []
    individual_f1s    = []
    ensemble_sizes    = []
    ensemble_accuracies = []
    ensemble_nlls     = []
    ensemble_eces     = []
    ensemble_f1s      = []
    
    # Ensemble-level early stopping initializations
    ensemble_no_improve = 0
    best_ensemble_acc = 0.0
    best_ensemble_nll = np.inf  # lower is better
    best_ensemble_ece = np.inf  # lower is better
    best_ensemble_f1  = 0.0      # higher is better

    # Begin ensemble training loop
    input_size = train_X.shape[1]
    for n in range(1, N + 1):
        print(f"\nTraining MLP {n}/{N}")
        
        model = MLP(input_size, output_size=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Individual model early stopping trackers
        best_val_loss = np.inf
        epochs_no_improve_model = 0
        best_model_state = None

        # Train current model for up to 'epochs' epochs with early stopping on validation loss
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                outputs = model(val_X.to(device))
                loss = criterion(outputs, val_y.to(device))
                val_loss = loss.item()

            print(f"Epoch {epoch+1} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve_model = 0
            else:
                epochs_no_improve_model += 1

            if epochs_no_improve_model >= model_patience:
                print(f"Early stopping for this model triggered at epoch {epoch+1}.")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        
        # Evaluation the individual model
        model.eval()
        with torch.no_grad():
            outputs = model(val_X.to(device))
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            acc, nll, ece, f1 = eval_metrics(val_y.numpy(), preds, probs, average='weighted')
            
            individual_accuracies.append(acc)
            individual_nlls.append(nll)
            individual_eces.append(ece)
            individual_f1s.append(f1)
        
        print(f"Model {n}: Accuracy = {acc:.4f}, NLL = {nll:.4f}, ECE = {ece:.4f}, F1 = {f1:.4f}")
        
        # After training, add current model to ensemble
        ensemble_models.append(model)
        
        # Evaluate the current ensemble
        acc, nll, ece, f1 = evaluate_ensemble(ensemble_models, val_X, val_y)
        ensemble_sizes.append(n)
        ensemble_accuracies.append(acc)
        ensemble_nlls.append(nll)
        ensemble_eces.append(ece)
        ensemble_f1s.append(f1)
        
        print(f"Ensemble Size {n} -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1: {f1:.4f}")
        
        # Determine whether ensemble performance has improved:
        # Improvement criteria:
        #   NLL and ECE decrease, and F1 increases.
        if nll < best_ensemble_nll or ece < best_ensemble_ece:
            best_ensemble_acc = acc
            best_ensemble_nll = nll
            best_ensemble_ece = ece
            best_ensemble_f1 = f1
            ensemble_no_improve = 0
            print("Ensemble performance improved (NLL or ECE decreased, ACC or F1 increased).")
        else:
            ensemble_no_improve += 1
            print(f"No ensemble improvement count: {ensemble_no_improve}/{ensemble_patience}")
        
        # Check ensemble-level early stopping condition
        if ensemble_no_improve >= ensemble_patience:
            print(f"Ensemble-level early stopping triggered after {n} models.")
            break
    
    ensemble_probs = evaluate_ensemble_for_plot(ensemble_models, val_X)
    
    # Optionally, you can plot the evolution of ensemble metrics here:
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 4, 1)
    plt.plot(ensemble_sizes, ensemble_accuracies, marker='o')
    plt.title('Ensemble Accuracy vs Ensemble Size')
    plt.xlabel('Number of MLPs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(ensemble_sizes, ensemble_nlls, marker='o', color='orange')
    plt.title('Ensemble NLL vs Ensemble Size')
    plt.xlabel('Number of MLPs')
    plt.ylabel('NLL')
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(ensemble_sizes, ensemble_eces, marker='o', color='green')
    plt.title('Ensemble ECE vs Ensemble Size')
    plt.xlabel('Number of MLPs')
    plt.ylabel('ECE')
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    plt.plot(ensemble_sizes, ensemble_f1s, marker='o', color='purple')
    plt.title('Ensemble F1 vs Ensemble Size')
    plt.xlabel('Number of MLPs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    individual_metrics = {
        'accuracies': individual_accuracies,
        'nlls': individual_nlls,
        'eces': individual_eces,
        'f1s': individual_f1s
    }
    
    # Return the ensemble models and the recorded metrics.
    metrics = {
        'ensemble_sizes': ensemble_sizes,
        'accuracies': ensemble_accuracies,
        'nlls': ensemble_nlls,
        'eces': ensemble_eces,
        'f1s': ensemble_f1s
    }
    
    
    return ensemble_models, metrics, ensemble_probs, individual_metrics

def train_student_model(StudentMLP, train_embeddings, train_labels, val_embeddings, val_labels, ensemble_models, 
                        epochs=50, T=2.0, alpha=0.7, patience=5):
    """
    Trains the student_model using knowledge distillation with early stopping based on
    validation F1 score. If validation F1 does not improve for 'patience' epochs,
    training is halted early and the best model (in terms of validation F1) is restored.
    
    Parameters:
        student_model: PyTorch model to be trained.
        train_loader: DataLoader for training data.
        val_X: Validation features as a Tensor (or will be moved to device).
        val_y: Validation labels as a numpy array (or tensor converted to numpy later).
        soft_targets: Soft targets from the teacher model (numpy array).
        epochs (int): Maximum number of epochs to train.
        T (float): Temperature parameter for distillation.
        alpha (float): Weighting between hard and soft target losses.
        patience (int): Number of consecutive epochs without improvement in F1 to wait before stopping.
    
    Returns:
        student_model: Trained student model (with best weights restored).
        loss_plot: List of training losses per epoch.
    """
    
    
    # Assume device is defined globally, or you can also pass it as a parameter.
    global device 
    
    train_X = torch.from_numpy(train_embeddings).float()
    train_y = torch.from_numpy(train_labels).long()
    val_X = torch.from_numpy(val_embeddings).float()
    val_y = torch.from_numpy(val_labels).long()

    # Prepare DataLoader with indices to match soft targets
    train_dataset_with_indices = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset_with_indices, batch_size=2048, shuffle=False)
    
    # Instantiate the student model
    student_model = StudentMLP(input_dim=train_X.shape[1], output_size=2)

    soft_targets = generate_soft_targets(ensemble_models, train_X)
    
    student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
    loss_plot = []
    
    # Lists to keep track of evaluation metrics on the validation set
    acc_list = []
    nll_list = []
    ece_list = []
    f1_list = []  # To store F1 score
    
    # Convert soft_targets to a tensor for easier batch extraction later
    soft_targets_tensor = torch.tensor(soft_targets, dtype=torch.float32)
    
    # Early stopping trackers
    best_f1 = -np.inf  # Higher F1 is better
    best_ece = np.inf
    best_nll = np.inf
    best_model_state = None
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        student_model.train()
        epoch_losses = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = student_model(inputs)  # Logits
            # Map the soft targets to the current batch using indices
            batch_start = batch_idx * train_loader.batch_size
            batch_end = batch_start + inputs.size(0)
            y_teacher_probs = soft_targets_tensor[batch_start:batch_end].to(device)
            
            # Compute the distillation loss
            loss = distillation_loss(outputs, labels, y_teacher_probs, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        loss_plot.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Set model to evaluation mode and compute metrics on validation set
        student_model.eval()
        with torch.no_grad():
            val_outputs = student_model(val_X.to(device))
            val_probs = nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=1)
            
            # Calculate Accuracy, NLL, ECE, and F1 score on validation data
            acc, nll, ece, f1 = eval_metrics(val_y.numpy(), val_preds, val_probs, average='weighted')
            
            acc_list.append(acc)
            nll_list.append(nll)
            ece_list.append(ece)
            f1_list.append(f1)
        
        print(f"Validation -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")
        
        # Check for improvement in F1 score
        if nll < best_nll or ece < best_ece:
            best_f1 = f1
            best_nll = nll
            best_ece = ece
            best_model_state = student_model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} consecutive epoch(s).")
        
        # If F1 hasn't improved for 'patience' epochs, stop training early.
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model.")
            if best_model_state is not None:
                student_model.load_state_dict(best_model_state)
            break

    # Plot the evaluation metrics
    epochs_range = range(1, len(loss_plot) + 1)
    plt.figure(figsize=(16, 4))
    
    # Accuracy Plot
    plt.subplot(1, 4, 1)
    plt.plot(epochs_range, acc_list, marker='o', color='blue')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # NLL Plot
    plt.subplot(1, 4, 2)
    plt.plot(epochs_range, nll_list, marker='o', color='green')
    plt.title('Validation NLL')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.grid(True)
    
    # ECE Plot
    plt.subplot(1, 4, 3)
    plt.plot(epochs_range, ece_list, marker='o', color='red')
    plt.title('Validation ECE')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.grid(True)
    
    # F1 Score Plot
    plt.subplot(1, 4, 4)
    plt.plot(epochs_range, f1_list, marker='o', color='purple')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(4, 4))
    # Loss Plot
    plt.plot(epochs_range, loss_plot, marker='o', color='orange')
    plt.title('Distillation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return student_model, loss_plot
