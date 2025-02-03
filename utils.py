import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.calibration import calibration_curve

# Check for GPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS device')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA device')
else:
    device = torch.device('cpu')
    print('No GPU found. Using CPU')

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def eval_metrics(true_labels, preds, probs, average='weighted'):
    """
    Evaluate classification metrics including accuracy, negative log-loss, 
    expected calibration error, and F1 score.

    Parameters:
    --------------------------
    true_labels: array-like
        The ground-truth labels.
    preds: array-like
        The predicted class labels.
    probs: array-like
        The predicted probabilities for each class.
    average: str, default='weighted'
        The type of averaging performed on the data when computing the F1 score. 
        This parameter is passed to the f1_score function from scikit-learn.

    Returns:
    --------------------------
    tuple
        A tuple containing:
        - acc: Accuracy score.
        - nll: Negative log loss.
        - ece: Expected Calibration Error.
        - f1: F1 score.
    """
    
    # Compute the accuracy score.
    acc = accuracy_score(true_labels, preds)
    
    # Compute the negative log loss.
    nll = log_loss(true_labels, probs) # Formula is
    
    # Compute the Expected Calibration Error using the external compute_ece function.
    ece = compute_ece(probs, true_labels)
    
    # Compute the F1 score using the provided averaging method.
    f1 = f1_score(true_labels, preds, average=average)
    
    return acc, nll, ece, f1

def plot_reliability_diagram(prob_pos, true_labels, n_bins=10):
    """
    Compute and plot a reliability diagram.
    
    Parameters:
    - prob_pos: 1D array of predicted probabilities for the positive class.
    - true_labels: 1D array of ground truth binary labels.
    - n_bins: Number of bins in the diagram.
    """
    # Define bins from 0 to 1
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Arrays to store the bin means and fraction of positives
    bin_preds = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Find samples with predictions in the current bin.
        indices = np.where((prob_pos >= bins[i]) & (prob_pos < bins[i+1]))[0]
        if len(indices) > 0:
            bin_preds[i] = np.mean(prob_pos[indices])
            bin_true[i] = np.mean(true_labels[indices])
            bin_counts[i] = len(indices)
    
    # Plot the reliability diagram
    plt.figure(figsize=(8, 6))
    plt.plot(bin_preds, bin_true, "s-", label="Ensemble")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.title("Reliability Diagram (Manual Binning)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.show()
    
    # Optionally print the bin counts for analysis
    print("Bin counts:", bin_counts)

def evaluate_ensemble(models, val_X, val_y):
    ensemble_outputs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            outputs = m(val_X.to(device))
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            ensemble_outputs.append(probs)
    ensemble_probs = np.mean(ensemble_outputs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    # eval_metrics should return: accuracy, nll, ece, f1
    acc, nll, ece, f1 = eval_metrics(val_y.numpy(), ensemble_preds, ensemble_probs, average='weighted')
    return acc, nll, ece, f1

def evaluate_ensemble_for_plot(models, val_X):
    ensemble_probs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            outputs = m(val_X.to(device))
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            ensemble_probs.append(probs)
    ensemble_probs = np.stack(ensemble_probs)
    ensemble_mean = np.mean(ensemble_probs, axis=0)
    ensemble_var = np.var(ensemble_probs, axis=0)
    ensemble_entropy = -np.sum(ensemble_mean * np.log(ensemble_mean + 1e-9), axis=1)
    
    ensemble_probs = {
    'probs': ensemble_probs,
    'mean': ensemble_mean,
    'var': ensemble_var,
    'entropy': ensemble_entropy
    }
    
    return ensemble_probs


def sst_process_embeddings_and_labels(
    train_embeddings_file='train_embeddings.npy',
    train_labels_file='train_labels.npy',
    validation_embeddings_file='validation_embeddings.npy',
    validation_labels_file='validation_labels.npy',
    test_embeddings_file='test_embeddings.npy',
    test_labels_file='test_labels.npy',
    model_checkpoint="HuggingFaceTB/SmolLM-1.7B-Instruct",
    dataset_name='glue',
    dataset_config='sst2',
    max_length=128,
    device=device
):
    """
    Load or compute embeddings and labels for the train, validation, and test splits.
    
    If the files exist, the embeddings and labels are loaded from disk.
    Otherwise, the function:
      1. Loads the tokenizer and model.
      2. Streams the specified dataset.
      3. Extracts embeddings from the last hidden state of the model.
      4. Standardizes the embeddings.
      5. Saves the embeddings and labels to disk.
      
    Parameters:
        train_embeddings_file (str): Path to save or load train embeddings.
        train_labels_file (str): Path to save or load train labels.
        validation_embeddings_file (str): Path to save or load validation embeddings.
        validation_labels_file (str): Path to save or load validation labels.
        test_embeddings_file (str): Path to save or load test embeddings.
        test_labels_file (str): Path to save or load test labels.
        model_checkpoint (str): Hugging Face model checkpoint.
        dataset_name (str): Name of the dataset to load.
        dataset_config (str): Configuration name of the dataset.
        max_length (int): Maximum sequence length for tokenization.
        device (torch.device): Device on which to run the model.
    
    Returns:
        tuple: A tuple containing:
            - train_embeddings (np.ndarray)
            - train_labels (np.ndarray)
            - validation_embeddings (np.ndarray)
            - validation_labels (np.ndarray)
            - test_embeddings (np.ndarray)
            - test_labels (np.ndarray)
    """
    
    # Check if all files exist
    if (os.path.exists(train_embeddings_file) and os.path.exists(train_labels_file) and
        os.path.exists(validation_embeddings_file) and os.path.exists(validation_labels_file) and
        os.path.exists(test_embeddings_file) and os.path.exists(test_labels_file)):
        
        # Load embeddings and labels from disk
        train_embeddings = np.load(train_embeddings_file)
        train_labels = np.load(train_labels_file)
        validation_embeddings = np.load(validation_embeddings_file)
        validation_labels = np.load(validation_labels_file)
        test_embeddings = np.load(test_embeddings_file)
        test_labels = np.load(test_labels_file)
        print("Loaded train, validation, and test embeddings and labels from disk.")
    
    else:
        # Initialize tokenizer and model from the Hugging Face model hub.
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        model.to(device)
        
        # Load the dataset using streaming to handle large datasets efficiently.
        dataset = load_dataset(dataset_name, dataset_config, streaming=True)
        
        # Initialize lists to store embeddings and labels.
        train_embeddings_list, train_labels_list = [], []
        validation_embeddings_list, validation_labels_list = [], []
        test_embeddings_list, test_labels_list = [], []
        
        def extract_embeddings(streamed_dataset, embeddings_list, labels_list):
            """
            Extract embeddings from dataset examples using the last layer hidden state.
            The embeddings are averaged over the sequence length.
            """
            for example in tqdm(streamed_dataset, desc="Processing examples"):
                # Tokenize the sentence from the dataset
                inputs = tokenizer(
                    example['sentence'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                # Forward pass through the model without gradient calculations.
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Extract the last hidden state and calculate the mean over the sequence length.
                hidden_states = outputs.hidden_states[-1]  # take the last layer
                embeddings = hidden_states.mean(dim=1).cpu().numpy()  # shape becomes (1, hidden_size)
                embeddings_list.append(embeddings.squeeze())  # remove redundant dimensions
                labels_list.append(example['label'])
        
        # Extract embeddings for each split.
        extract_embeddings(dataset['train'], train_embeddings_list, train_labels_list)
        extract_embeddings(dataset['validation'], validation_embeddings_list, validation_labels_list)
        extract_embeddings(dataset['test'], test_embeddings_list, test_labels_list)
        
        # Convert lists to numpy arrays for further processing.
        train_embeddings = np.vstack(train_embeddings_list)
        train_labels = np.array(train_labels_list)
        validation_embeddings = np.vstack(validation_embeddings_list)
        validation_labels = np.array(validation_labels_list)
        test_embeddings = np.vstack(test_embeddings_list)
        test_labels = np.array(test_labels_list)
        
        # Standardize the embeddings using StandardScaler.
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        validation_embeddings = scaler.transform(validation_embeddings)
        test_embeddings = scaler.transform(test_embeddings)
        
        # Save the processed data to disk for future runs.
        np.save(train_embeddings_file, train_embeddings)
        np.save(train_labels_file, train_labels)
        np.save(validation_embeddings_file, validation_embeddings)
        np.save(validation_labels_file, validation_labels)
        np.save(test_embeddings_file, test_embeddings)
        np.save(test_labels_file, test_labels)
        
        print("Saved train, validation, and test embeddings and labels to disk.")
    
    # Log the shapes of the embeddings and labels.
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation embeddings shape: {validation_embeddings.shape}")
    print(f"Validation labels shape: {validation_labels.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    return (train_embeddings, train_labels,
            validation_embeddings, validation_labels,
            test_embeddings, test_labels)

def imdb_process_embeddings_and_labels(
    imdb_train_embeddings_file='imdb_long_train_em.npy',
    imdb_train_labels_file='imdb_long_train_la.npy',
    imdb_test_embeddings_file='imdb_long_test_em.npy',
    imdb_test_labels_file='imdb_long_test_la.npy',
    imdb_unsupervised_embeddings_file='imdb_long_unsuper_em.npy',
    imdb_unsupervised_labels_file='imdb_long_unsuper_la.npy',
    model_checkpoint="HuggingFaceTB/SmolLM-1.7B-Instruct",
    dataset_name="stanfordnlp/imdb",
    max_length=512,
    device=device
):
    """
    Load or compute embeddings and labels for the train, validation, and test splits.
    
    If the files exist, the embeddings and labels are loaded from disk.
    Otherwise, the function:
      1. Loads the tokenizer and model.
      2. Streams the specified dataset.
      3. Extracts embeddings from the last hidden state of the model.
      4. Standardizes the embeddings.
      5. Saves the embeddings and labels to disk.
      
    Parameters:
        train_embeddings_file (str): Path to save or load train embeddings.
        train_labels_file (str): Path to save or load train labels.
        validation_embeddings_file (str): Path to save or load validation embeddings.
        validation_labels_file (str): Path to save or load validation labels.
        test_embeddings_file (str): Path to save or load test embeddings.
        test_labels_file (str): Path to save or load test labels.
        model_checkpoint (str): Hugging Face model checkpoint.
        dataset_name (str): Name of the dataset to load.
        dataset_config (str): Configuration name of the dataset.
        max_length (int): Maximum sequence length for tokenization.
        device (torch.device): Device on which to run the model.
    
    Returns:
        tuple: A tuple containing:
            - train_embeddings (np.ndarray)
            - train_labels (np.ndarray)
            - validation_embeddings (np.ndarray)
            - validation_labels (np.ndarray)
            - test_embeddings (np.ndarray)
            - test_labels (np.ndarray)
    """
    
    # Check if all files exist
    if (os.path.exists(imdb_train_embeddings_file) and os.path.exists(imdb_train_labels_file) and
        os.path.exists(imdb_test_embeddings_file) and os.path.exists(imdb_test_labels_file) and
        os.path.exists(imdb_unsupervised_embeddings_file) and os.path.exists(imdb_unsupervised_labels_file)):
        
        # Load embeddings and labels from disk
        imdb_train_embeddings = np.load(imdb_train_embeddings_file)
        imdb_train_labels = np.load(imdb_train_labels_file)
        imdb_test_embeddings = np.load(imdb_test_embeddings_file)
        imdb_test_labels = np.load(imdb_test_labels_file)
        imdb_unsupervised_embeddings = np.load(imdb_unsupervised_embeddings_file)
        imdb_unsupervised_labels = np.load(imdb_unsupervised_labels_file)
        print("Loaded train, test, and unsupervised embeddings and labels from disk.")
    
    else:
        # Initialize tokenizer and model from the Hugging Face model hub.
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        model.to(device)
        
        # Load the dataset using streaming to handle large datasets efficiently.
        dataset = load_dataset(dataset_name, streaming=True)
        
        # Initialize lists to store embeddings and labels.
        imdb_train_embeddings_list, imdb_train_labels_list = [], []
        imdb_test_embeddings_list, imdb_test_labels_list = [], []
        imdb_unsupervised_embeddings_list, imdb_unsupervised_labels_list = [], []
        
        def extract_embeddings(streamed_dataset, embeddings_list, labels_list):
            """
            Extract embeddings from dataset examples using the last layer hidden state.
            The embeddings are averaged over the sequence length.
            """
            for example in tqdm(streamed_dataset, desc="Processing examples"):
                # Tokenize the sentence from the dataset
                inputs = tokenizer(
                    example['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                # Forward pass through the model without gradient calculations.
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Extract the last hidden state and calculate the mean over the sequence length.
                hidden_states = outputs.hidden_states[-1]  # take the last layer
                embeddings = hidden_states.mean(dim=1).cpu().numpy()  # shape becomes (1, hidden_size)
                embeddings_list.append(embeddings.squeeze())  # remove redundant dimensions
                labels_list.append(example['label'])
        
        # Extract embeddings for each split.
        extract_embeddings(dataset['train'], imdb_train_embeddings_list, imdb_train_labels_list)
        extract_embeddings(dataset['test'], imdb_test_embeddings_list, imdb_test_labels_list)
        extract_embeddings(dataset['unsupervised'], imdb_unsupervised_embeddings_list, imdb_unsupervised_labels_list)

        # Convert lists to numpy arrays for further processing.
        imdb_train_embeddings = np.vstack(imdb_train_embeddings_list)
        imdb_train_labels = np.array(imdb_train_labels_list)
        imdb_test_embeddings = np.vstack(imdb_test_embeddings_list)
        imdb_test_labels = np.array(imdb_test_labels_list)
        imdb_unsupervised_embeddings = np.vstack(imdb_unsupervised_embeddings_list)
        imdb_unsupervised_labels = np.array(imdb_unsupervised_labels_list)
        
        # Standardize the embeddings using StandardScaler.
        scaler = StandardScaler()
        imdb_train_embeddings = scaler.fit_transform(imdb_train_embeddings)
        imdb_test_embeddings = scaler.transform(imdb_test_embeddings)
        imdb_unsupervised_embeddings = scaler.transform(imdb_unsupervised_embeddings)
        
        # Save the processed data to disk for future runs.
        np.save(imdb_train_embeddings_file, imdb_train_embeddings)
        np.save(imdb_train_labels_file, imdb_train_labels)
        np.save(imdb_test_embeddings_file, imdb_test_embeddings)
        np.save(imdb_test_labels_file, imdb_test_labels)
        np.save(imdb_unsupervised_embeddings_file, imdb_unsupervised_embeddings)
        np.save(imdb_unsupervised_labels_file, imdb_unsupervised_labels)
        
        print("Saved train, test, and unsupervised embeddings and labels to disk.")
    
    # Log the shapes of the embeddings and labels.
    print(f"IMDB Train embeddings shape: {imdb_train_embeddings.shape}")
    print(f"IMDB Train labels shape: {imdb_train_labels.shape}")
    print(f"IMDB Test embeddings shape: {imdb_test_embeddings.shape}")
    print(f"IMDB Test labels shape: {imdb_test_labels.shape}")
    print(f"IMDB Unsupervised embeddings shape: {imdb_unsupervised_embeddings.shape}")
    print(f"IMDB Unsupervised labels shape: {imdb_unsupervised_labels.shape}")
    
    return (imdb_train_embeddings, imdb_train_labels,
            imdb_test_embeddings, imdb_test_labels,
            imdb_unsupervised_embeddings, imdb_unsupervised_labels)



# def evaluate_models(test_embeddings, test_labels, student_model, ensemble_models, device, batch_size=32):
#     """
#     Evaluates three models on the given test dataset:
#     1. Student model.
#     2. An individual model (first model from the ensemble).
#     3. The ensemble (averaged softmax probabilities from all models in ensemble_models).

#     Parameters:
#     - test_embeddings (np.ndarray): The embeddings for the test dataset.
#     - test_labels (np.ndarray): The true labels for the test dataset.
#     - student_model (torch.nn.Module): The student model to evaluate.
#     - ensemble_models (list of torch.nn.Module): A list of models constituting the ensemble.
#     - device (torch.device): The device on which computations are performed.
#     - batch_size (int): Batch size used for DataLoader (default: 32).

#     Returns:
#     - metrics_dict (dict): A dictionary containing the evaluation metrics for
#       'student', 'individual', and 'ensemble' models.
#       Each entry is a tuple: (accuracy, nll, ece, f1)
#     """

#     # Convert embeddings and labels to tensors.
#     test_X = torch.from_numpy(test_embeddings).float()
#     test_y = torch.from_numpy(test_labels).long()

#     # Optionally create a DataLoader (if batch processing is desired)
#     test_dataset = TensorDataset(test_X, test_y)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Convert true labels to a NumPy array for eval_metrics function
#     true_labels = test_y.numpy()

#     # Dictionary to store metrics for each model.
#     metrics_dict = {}

#     # -------------------------
#     # Evaluation for Student Model
#     # -------------------------
#     student_model.to(device)
#     student_model.eval()
#     with torch.no_grad():
#         # Here, we run inference on the entire test_X.
#         outputs = student_model(test_X.to(device))
#         # Apply softmax to obtain probabilities
#         probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
#         # Compute predicted classes
#         preds = np.argmax(probs, axis=1)
        
#         # Compute evaluation metrics (you need to have an eval_metrics function defined)
#         acc, nll, ece, f1 = eval_metrics(true_labels, preds, probs, average='weighted')
#         metrics_dict['student'] = (acc, nll, ece, f1)

#     # Print student model metrics (optional)
#     print(f"Student -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

#     # -------------------------
#     # Evaluation for an Individual Model (first model in ensemble_models)
#     # -------------------------
#     individual_model = ensemble_models[0]
#     individual_model.to(device)
#     individual_model.eval()
#     with torch.no_grad():
#         outputs = individual_model(test_X.to(device))
#         probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
#         preds = np.argmax(probs, axis=1)
        
#     acc, nll, ece, f1 = eval_metrics(true_labels, preds, probs, average='weighted')
#     metrics_dict['individual'] = (acc, nll, ece, f1)

#     print(f"Individual -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

#     # -------------------------
#     # Evaluation for the Ensemble Model
#     # -------------------------
#     ensemble_outputs = []
#     for model in ensemble_models:
#         model.to(device)
#         model.eval()
#         with torch.no_grad():
#             outputs = model(test_X.to(device))
#             probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
#             ensemble_outputs.append(probs)

#     # Average the probabilities from all ensemble models.
#     ensemble_probs = np.mean(ensemble_outputs, axis=0)
#     ensemble_preds = np.argmax(ensemble_probs, axis=1)

#     acc, nll, ece, f1 = eval_metrics(true_labels, ensemble_preds, ensemble_probs, average='weighted')
#     metrics_dict['ensemble'] = (acc, nll, ece, f1)

#     print(f"Ensemble -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

#     return metrics_dict

# ------------------------------------------------------------------------------
# Define a consistent color mapping for the models.
# ------------------------------------------------------------------------------
model_colors = {
    'Student': 'red',
    'Individual': 'green',
    'Ensemble': 'blue'
}

# ------------------------------------------------------------------------------
# Plotting Helper Functions (with Dataset Information)
# ------------------------------------------------------------------------------

def plot_reliability_diagram_multi(true_labels, probs_dict, model_colors, n_bins=10, dataset_title="", trained_dataset=""):
    """
    Plots a reliability diagram for multiple models on the same plot,
    using a consistent color for each model.
    
    The title includes the evaluation dataset and the training dataset.
    """
    plt.figure(figsize=(6,6))
    for model_name, probs in probs_dict.items():
        # For binary classification, assume positive class is at index 1.
        positive_probs = probs[:, 1]
        prob_true, prob_pred = calibration_curve(true_labels, positive_probs, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name, color=model_colors[model_name])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f"Reliability Diagram ({dataset_title}) - Trained on {trained_dataset}")
    plt.legend()
    plt.show()

def plot_confidence_histogram_side_by_side(probs_dict, model_colors, bins=20, dataset_title="", trained_dataset=""):
    """
    Plots separate confidence histograms side-by-side for each model,
    each with a consistent color. The overall figure title includes dataset info.
    """
    model_names = list(probs_dict.keys())
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 4))
    
    if num_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        confidences = np.max(probs_dict[model_name], axis=1)
        ax.hist(confidences, bins=bins, density=True, color=model_colors[model_name], alpha=0.7)
        ax.set_title(f"{model_name} Confidence")
        ax.set_xlabel("Max Softmax Probability")
        ax.set_ylabel("Density")
    fig.suptitle(f"Confidence Histogram ({dataset_title}) - Trained on {trained_dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def plot_entropy_histogram_side_by_side(probs_dict, model_colors, bins=20, dataset_title="", trained_dataset=""):
    """
    Plots separate entropy histograms side-by-side for each model,
    each with a consistent color. The overall figure title includes dataset info.
    
    Predictive entropy is computed as:
        H(p) = -sum(p * log(p + epsilon))
    """
    epsilon = 1e-12
    model_names = list(probs_dict.keys())
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 4))
    
    if num_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        entropies = -np.sum(probs_dict[model_name] * np.log(probs_dict[model_name] + epsilon), axis=1)
        ax.hist(entropies, bins=bins, density=True, color=model_colors[model_name], alpha=0.7)
        ax.set_title(f"{model_name} Entropy")
        ax.set_xlabel("Predictive Entropy")
        ax.set_ylabel("Density")
    fig.suptitle(f"Entropy Histogram ({dataset_title}) - Trained on {trained_dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def compute_mutual_information(ensemble_outputs, ensemble_probs):
    """
    Computes the mutual information (MI) for each sample given an ensemble.
    
    MI = H(ensemble average) - average_i H(model_i)
    """
    epsilon = 1e-12
    H_ensemble = -np.sum(ensemble_probs * np.log(ensemble_probs + epsilon), axis=1)
    individual_entropies = []
    for probs in ensemble_outputs:
        H_ind = -np.sum(probs * np.log(probs + epsilon), axis=1)
        individual_entropies.append(H_ind)
    individual_entropies = np.stack(individual_entropies, axis=0)
    avg_ind_entropy = np.mean(individual_entropies, axis=0)
    MI = H_ensemble - avg_ind_entropy
    return MI

def plot_mutual_information(MI, bins=20, dataset_title="", trained_dataset=""):
    """
    Plots the histogram of mutual information (MI) values.
    """
    plt.figure(figsize=(8,4))
    plt.hist(MI, bins=bins, alpha=0.7, color='purple', density=True)
    plt.xlabel('Mutual Information')
    plt.ylabel('Density')
    plt.title(f"Mutual Information Histogram ({dataset_title}) - Trained on {trained_dataset}")
    plt.show()

def plot_ensemble_variance(ensemble_outputs, bins=20, dataset_title="", trained_dataset=""):
    """
    Computes and plots the mean variance of the ensemble predictions for each sample.
    
    The variance is computed across models for each class and then averaged over classes.
    """
    ensemble_outputs_arr = np.stack(ensemble_outputs, axis=0)
    variance_per_sample = np.var(ensemble_outputs_arr, axis=0)
    mean_variance = np.mean(variance_per_sample, axis=1)
    plt.figure(figsize=(8,4))
    plt.hist(mean_variance, bins=bins, alpha=0.7, color='orange', density=True)
    plt.xlabel('Mean Variance across Ensemble Models')
    plt.ylabel('Density')
    plt.title(f"Ensemble Variance Histogram ({dataset_title}) - Trained on {trained_dataset}")
    plt.show()

# ------------------------------------------------------------------------------
# Evaluation Function (with Dataset and Training Info)
# ------------------------------------------------------------------------------

def evaluate_models(test_embeddings, test_labels, student_model, ensemble_models, device, batch_size=32, dataset_key="test_em", trained_dataset="SST2"):
    """
    Evaluates three models on the given test dataset and generates uncertainty plots,
    while displaying the evaluation dataset name and the training dataset in each graph.
    
    Models evaluated:
      1. Student model.
      2. An individual model (first model from the ensemble).
      3. The ensemble (averaged softmax probabilities from all ensemble models).
    
    The following combined plots are generated:
      - Reliability Diagram (with dataset info)
      - Side-by-side Confidence Histograms (with dataset info)
      - Side-by-side Entropy Histograms (with dataset info)
      
    For the ensemble, additional plots are produced:
      - Ensemble Variance Histogram
      - Mutual Information Histogram
    
    Parameters:
      - test_embeddings (np.ndarray): Embeddings for the test dataset.
      - test_labels (np.ndarray): True labels for the test dataset.
      - student_model (torch.nn.Module): Student model.
      - ensemble_models (list of torch.nn.Module): List of ensemble models.
      - device (torch.device): Device for computation.
      - batch_size (int): Batch size for DataLoader.
      - dataset_key (str): Key identifying the evaluation dataset (e.g., "imdb_train_em", "test_em").
      - trained_dataset (str): Indicates the dataset the models were trained on (e.g., "SST2" or "IMDB").
    
    Returns:
      - metrics_dict (dict): Dictionary with evaluation metrics for 'student', 'individual', and 'ensemble'.
                             Each entry is a tuple: (accuracy, nll, ece, f1)
    
    Note: This function assumes an external `eval_metrics` function is defined that takes
          (true_labels, predictions, probabilities, average=...) and returns the desired metrics.
    """
    # Mapping from dataset_key to friendly title.
    dataset_names_map = {
        "imdb_train": "IMDB train subset",
        "imdb_test": "IMDB test subset",
        "sst2_train": "SST2 train subset",
        "sst2_test": "SST2 test subset",
        "cola_train": "CoLa train subset",
        "cola_test": "CoLa test subset"
    }
    dataset_title = dataset_names_map.get(dataset_key, dataset_key)
    
    # Convert embeddings and labels to tensors.
    test_X = torch.from_numpy(test_embeddings).float()
    test_y = torch.from_numpy(test_labels).long()
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # True labels as numpy array for evaluation and plotting.
    true_labels = test_y.numpy()
    
    metrics_dict = {}
    probs_dict = {}  # To store probability outputs for combined plotting.
    
    # -------------------------
    # 1. Student Model Evaluation
    # -------------------------
    student_model.to(device)
    student_model.eval()
    with torch.no_grad():
        outputs = student_model(test_X.to(device))
        probs_student = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        preds_student = np.argmax(probs_student, axis=1)
        acc, nll, ece, f1 = eval_metrics(true_labels, preds_student, probs_student, average='weighted')
        metrics_dict['student'] = (acc, nll, ece, f1)
        probs_dict['Student'] = probs_student
    print(f"Student -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")
    
    # -------------------------
    # 2. Individual Model Evaluation (first model from the ensemble)
    # -------------------------
    individual_model = ensemble_models[0]
    individual_model.to(device)
    individual_model.eval()
    with torch.no_grad():
        outputs = individual_model(test_X.to(device))
        probs_individual = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        preds_individual = np.argmax(probs_individual, axis=1)
    acc, nll, ece, f1 = eval_metrics(true_labels, preds_individual, probs_individual, average='weighted')
    metrics_dict['individual'] = (acc, nll, ece, f1)
    probs_dict['Individual'] = probs_individual
    print(f"Individual -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")
    
    # -------------------------
    # 3. Ensemble Model Evaluation
    # -------------------------
    ensemble_outputs = []  # To store probability outputs from each ensemble member.
    for model in ensemble_models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(test_X.to(device))
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            ensemble_outputs.append(probs)
    
    # Average the probabilities from all ensemble models.
    ensemble_probs = np.mean(ensemble_outputs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    acc, nll, ece, f1 = eval_metrics(true_labels, ensemble_preds, ensemble_probs, average='weighted')
    metrics_dict['ensemble'] = (acc, nll, ece, f1)
    probs_dict['Ensemble'] = ensemble_probs
    print(f"Ensemble -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")
    
    # ------------------------------------------------------------------------------
    # Generate Combined Plots with Dataset and Training Info
    # ------------------------------------------------------------------------------
    plot_reliability_diagram_multi(true_labels, probs_dict, model_colors, n_bins=10,
                                   dataset_title=dataset_title, trained_dataset=trained_dataset)
    plot_confidence_histogram_side_by_side(probs_dict, model_colors, bins=20,
                                           dataset_title=dataset_title, trained_dataset=trained_dataset)
    plot_entropy_histogram_side_by_side(probs_dict, model_colors, bins=20,
                                        dataset_title=dataset_title, trained_dataset=trained_dataset)
    
    # ------------------------------------------------------------------------------
    # Additional Ensemble-Specific Uncertainty Plots
    # ------------------------------------------------------------------------------
    plot_ensemble_variance(ensemble_outputs, bins=20,
                           dataset_title=dataset_title, trained_dataset=trained_dataset)
    MI = compute_mutual_information(ensemble_outputs, ensemble_probs)
    plot_mutual_information(MI, bins=20,
                            dataset_title=dataset_title, trained_dataset=trained_dataset)
    
    return metrics_dict




def generate_soft_targets(ensemble_models, train_X):
    all_probs = []
    train_X = train_X.to(device)
    for model in ensemble_models:
        model.eval()
        model.to(device)
        with torch.no_grad():
            outputs = model(train_X)
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
        model.cpu()
    all_probs = np.stack(all_probs)  # Shape: (N_models, N_samples, N_classes)
    soft_targets = np.mean(all_probs, axis=0)
    return soft_targets


def distillation_loss(y_student_logits, y_true_labels, y_teacher_probs, T=2.0, alpha=0.7):
    """
    y_student_logits: Logits output by the student model
    y_true_labels: Ground truth labels
    y_teacher_probs: Soft targets (probabilities) from the teacher (ensemble)
    T: Temperature parameter
    alpha: Weighting factor between soft and hard targets
    """
    # Cross-Entropy Loss with true labels (hard targets)
    ce_loss = nn.CrossEntropyLoss()(y_student_logits, y_true_labels)
    
    # KL Divergence Loss with soft targets (from teacher)
    log_student_probs = nn.functional.log_softmax(y_student_logits / T, dim=1)
    teacher_probs_T = torch.tensor(y_teacher_probs, dtype=torch.float32).to(y_student_logits.device)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(log_student_probs, teacher_probs_T)
    
    # Combined Loss
    loss = alpha * ce_loss + (1 - alpha) * (T ** 2) * kl_loss
    # loss = alpha
    return loss