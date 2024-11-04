Målet er at undersøge om vi kan forbedre usikkerhed på data via en pre-trained model, som vi trækker ned. Undersøge 10 samples og sammenlign.

Kan vi tage disse samples og distillere dem ned til en model, som kan forbedre usikkerheden på data?

Finde et classification problem at arbejde med. Finde om det allerede er lavet.

$$ D =\{(x_i, y_i)\}_{i=1}^N $$

$$ D \rightarrow \text{LLM}^{\mathbb{Z}_i} \rightarrow \text{MLP} \rightarrow \hat{y}_i \rightarrow \text{Ensample} \rightarrow y^{(E)}_i$$

ECE: Expected Calibration Error

Til næste gang:

- Finde model
- Finde datasæt
- Forstå ECE bedre
- Tidsplan

Datasæt: https://huggingface.co/datasets/stanfordnlp/sst2

Mulige spørgsmål:

- Kan vi få en bedre performance på usikkerhed? (distillering)
- Kan vi kombinere ensambles tilbage til en model?

Finetune distilBERT på SST-2 og sammenlign med BERT?

Projektplan:

- første udgave af introduktion
- databeskrivelse
- modelbeskrivelse
- metode


Var ensemble flere forskellige llm
MLP?




# Create DataLoader for training data with shuffling
train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam

# Train the ensemble of MLPs


for model in ensemble:
    model.train()
    model_optimizer = optimizer(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_embeddings, batch_labels in train_loader:
            # Forward pass
            logits = model(batch_embeddings)
            loss = criterion(logits, batch_labels)
            # Backward pass
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
        print(f"Model {ensemble.index(model)+1} Epoch {epoch+1} Loss: {loss.item()}")
    print()

# Evaluate the ensemble on the validation set
ensemble_predictions = []
for model in ensemble:
    model.eval()
    with torch.no_grad():
        logits = model(validation_embeddings_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        ensemble_predictions.append(predictions)
        
# Combine predictions from all models
ensemble_predictions = np.vstack(ensemble_predictions)
ensemble_predictions = np.transpose(ensemble_predictions)  # shape (num_examples, ensemble_size)

# Compute the final prediction by majority voting
final_predictions = np.zeros(ensemble_predictions.shape[0])
for i, predictions in enumerate(ensemble_predictions):
    final_predictions[i] = np.argmax(np.bincount(predictions))
    
# Compute accuracy
accuracy = accuracy_score(validation_labels, final_predictions)
print(f"Ensemble Accuracy: {accuracy}")