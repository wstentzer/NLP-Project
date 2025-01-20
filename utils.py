import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score

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
    
    Notes:
    --------------------------
    - This function assumes that the `compute_ece` function is defined or imported from an external package.
    - If `compute_ece` is not yet defined, you will have to implement this function or find an appropriate library.
    """
    # Compute the accuracy score.
    acc = accuracy_score(true_labels, preds)
    
    # Compute the negative log loss.
    nll = log_loss(true_labels, probs)
    
    # Compute the Expected Calibration Error using the external compute_ece function.
    ece = compute_ece(probs, true_labels)
    
    # Compute the F1 score using the provided averaging method.
    f1 = f1_score(true_labels, preds, average=average)
    
    return acc, nll, ece, f1

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

def evaluate_models(test_embeddings, test_labels, student_model, ensemble_models, device, batch_size=32):
    """
    Evaluates three models on the given test dataset:
    1. Student model.
    2. An individual model (first model from the ensemble).
    3. The ensemble (averaged softmax probabilities from all models in ensemble_models).

    Parameters:
    - test_embeddings (np.ndarray): The embeddings for the test dataset.
    - test_labels (np.ndarray): The true labels for the test dataset.
    - student_model (torch.nn.Module): The student model to evaluate.
    - ensemble_models (list of torch.nn.Module): A list of models constituting the ensemble.
    - device (torch.device): The device on which computations are performed.
    - batch_size (int): Batch size used for DataLoader (default: 32).

    Returns:
    - metrics_dict (dict): A dictionary containing the evaluation metrics for
      'student', 'individual', and 'ensemble' models.
      Each entry is a tuple: (accuracy, nll, ece, f1)
    """

    # Convert embeddings and labels to tensors.
    test_X = torch.from_numpy(test_embeddings).float()
    test_y = torch.from_numpy(test_labels).long()

    # Optionally create a DataLoader (if batch processing is desired)
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Convert true labels to a NumPy array for eval_metrics function
    true_labels = test_y.numpy()

    # Dictionary to store metrics for each model.
    metrics_dict = {}

    # -------------------------
    # Evaluation for Student Model
    # -------------------------
    student_model.to(device)
    student_model.eval()
    with torch.no_grad():
        # Here, we run inference on the entire test_X.
        outputs = student_model(test_X.to(device))
        # Apply softmax to obtain probabilities
        probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        # Compute predicted classes
        preds = np.argmax(probs, axis=1)
        
        # Compute evaluation metrics (you need to have an eval_metrics function defined)
        acc, nll, ece, f1 = eval_metrics(true_labels, preds, probs, average='weighted')
        metrics_dict['student'] = (acc, nll, ece, f1)

    # Print student model metrics (optional)
    print(f"Student -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

    # -------------------------
    # Evaluation for an Individual Model (first model in ensemble_models)
    # -------------------------
    individual_model = ensemble_models[0]
    individual_model.to(device)
    individual_model.eval()
    with torch.no_grad():
        outputs = individual_model(test_X.to(device))
        probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
    acc, nll, ece, f1 = eval_metrics(true_labels, preds, probs, average='weighted')
    metrics_dict['individual'] = (acc, nll, ece, f1)

    print(f"Individual -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

    # -------------------------
    # Evaluation for the Ensemble Model
    # -------------------------
    ensemble_outputs = []
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

    print(f"Ensemble -- Accuracy: {acc:.4f}, NLL: {nll:.4f}, ECE: {ece:.4f}, F1 Score: {f1:.4f}")

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