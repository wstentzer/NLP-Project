Below is an **undergraduate-friendly methodology** in paragraph form, integrating both **formulae** and their **explanations** within the text. The methodology covers two mirrored experiments: in **Experiment A**, we train on **SST2** and treat **IMDB** as out-of-distribution (OOD), and in **Experiment B**, we train on **IMDB** and treat **SST2** as OOD.

---

### Overview

We explore how **Multilayer Perceptrons (MLPs)** perform when trained and tested on two well-known sentiment analysis datasets, **SST2** and **IMDB**. Each experiment involves typical steps of **data preprocessing**, **model training**, **ensembling**, and—importantly—an **out-of-distribution** check. Specifically, in **Experiment A**, we train on SST2 and use IMDB to test generalization. In **Experiment B**, we reverse this procedure: we train on IMDB and use SST2 as an OOD test. Throughout this process, we measure metrics such as **Accuracy**, **Negative Log-Likelihood (NLL)**, **Expected Calibration Error (ECE)**, and **F1-Score**, and we employ **knowledge distillation** to compress ensembles into a single student model.

---

### Data Preprocessing

All text data is first **tokenized** into word pieces or subwords, ensuring each sample is converted into a sequence of numeric indices. Because **SST2** reviews are short, we set a maximum token length of **128**, while **IMDB** reviews can be much longer, so we set **1024** as the maximum. Any text exceeding these lengths is truncated, and shorter texts are padded to maintain a fixed length.

Once embedded (i.e., mapped into numeric vectors), we **standardize** each feature dimension by subtracting the mean and dividing by the standard deviation computed over the **training split**. Formally, for a feature \(x\),

\[
x' = \frac{x - \mu}{\sigma},
\]

where \(x'\) is the standardized feature, \(\mu\) is the mean of \(x\) in the training data, and \(\sigma\) is the standard deviation. This process enhances numerical stability and convergence speed for subsequent MLP training, and we apply the same \(\mu\) and \(\sigma\) to the validation/test and OOD sets to avoid data leakage.

---

### Model Architecture and Training

We adopt a straightforward **MLP architecture**, starting with an **input layer** that takes the standardized embedding vectors. We follow this with one or more **fully connected (dense) layers**—often with dropout to mitigate overfitting—and conclude with an **output layer** that yields a single value (the **logit**). We do **not** apply a Sigmoid or Softmax within the model itself; instead, we rely on specialized loss functions that handle the activation step internally.

For binary classification, we use **binary cross-entropy (BCE)** with logits. In standard form, the BCE for a single sample is:

\[
\text{BCE}( \hat{y}, y ) 
= - \Big[ y \log(\sigma(z)) + (1 - y) \log\bigl(1 - \sigma(z)\bigr) \Big],
\]

where \(y \in \{0, 1\}\) is the true label, \(z\) is the raw logit from the MLP, \(\sigma(\cdot)\) is the Sigmoid function, and \(\hat{y} = \sigma(z)\). In frameworks like PyTorch, functions such as `BCEWithLogitsLoss` merge the Sigmoid and BCE calculations for greater numerical stability. We use **Adam** with a learning rate of \(1 \times 10^{-3}\) and a relatively large **batch size** (often 2048), stopping early if the validation loss plateaus.

---

### Ensemble Learning

To form an **ensemble**, we train multiple MLPs (each initialized randomly and often presented with shuffled data) on the same training set. Although each model learns the same overall task, they can converge to slightly different “solutions,” improving robustness when combined. After training, each MLP produces a probability prediction \(\hat{y}_k\). We then compute an **ensemble probability** by averaging across all \(K\) models:

\[
\hat{y}_{\text{ensemble}} = \frac{1}{K} \sum_{k=1}^K \hat{y}_k.
\]

For binary sentiment classification, if \(\hat{y}_{\text{ensemble}}>0.5\), we predict the positive class; otherwise, we predict the negative class. Alongside typical early stopping for training individual MLPs, we can also decide how many models to include in the ensemble by checking whether adding another model significantly lowers metrics like **Negative Log-Likelihood (NLL)** or **Expected Calibration Error (ECE)** on the validation set. If the improvement is negligible, we stop expanding the ensemble.

---

### Knowledge Distillation

While ensembling enhances accuracy and calibration, it can be computationally expensive at inference time. To address this, we employ **knowledge distillation**, where we train a single **student MLP** to mimic the ensemble’s output probabilities. First, we gather **soft targets** by passing each training sample through the ensemble and averaging the resulting probabilities. We then combine these soft targets with the original hard labels \(y\) in a **distillation loss**:

\[
\mathcal{L}_{\text{distill}} 
= \alpha \,\mathcal{L}_{\text{CE}} \bigl( p_{\text{student}},\, y \bigr) 
\;+\; (1 - \alpha)\,T^2 \,\mathrm{KL}\!\Bigl( p_{\text{student}}^T ,\, p_{\text{ensemble}}^T \Bigr),
\]

where:
- \(\alpha\in[0,1]\) balances the importance of the real labels vs. the ensemble’s soft labels.  
- \(T\) is a **temperature** parameter that “softens” or “sharpens” the probability distributions \(p_{\text{student}}^T\) and \(p_{\text{ensemble}}^T\).  
- \(\mathrm{KL}(\cdot || \cdot)\) is the **Kullback–Leibler divergence**, measuring how one probability distribution differs from another.

By minimizing KL divergence, the student learns to match the ensemble’s detailed probability patterns. The end result is a single model that often achieves accuracy close to the ensemble with less computational overhead.

---

### Experiment A: Train on SST2, Use IMDB for OOD Testing

In the first experiment, we train our MLPs on the **SST2** dataset, applying the above steps of tokenization (max length 128), standardization, and MLP training with BCE. We split SST2 into **train/validation/test** subsets. After finalizing each model (and optionally creating a student model via distillation), we evaluate them on **SST2’s test set** using **Accuracy**, **NLL**, **ECE**, and **F1-Score**.

Next, to assess generalization, we apply the same trained models to the **IMDB** dataset (max length 1024). Since IMDB reviews differ in length and detail from SST2 snippets, we treat IMDB as an **out-of-distribution (OOD)** dataset. Any performance drop compared to in-domain testing indicates how well the SST2-trained ensemble or student model adapts to a domain shift.

---

### Experiment B: Train on IMDB, Use SST2 for OOD Testing

In the second experiment, we reverse the roles. We train on **IMDB**, where each review can be up to 1024 tokens, following the same procedure: tokenization, standardization, MLP training with BCE, and (if desired) knowledge distillation to compress the ensemble. Again, we split IMDB into **train/validation/test** to guide optimization and early stopping.

Once satisfied with the IMDB-trained models, we evaluate them on the **IMDB test** set to see how well they perform in-domain. We then use **SST2** as the **OOD** dataset, since its shorter reviews differ notably from IMDB’s longer texts. By comparing metrics such as Accuracy, NLL, ECE, and F1-Score on SST2 vs. IMDB, we gain insight into how robustly an IMDB-trained system handles short-review sentiment classification.

---

### Evaluation Metrics

Throughout both experiments, we track four main metrics. First, **Accuracy** captures the proportion of correct predictions. Next, **Negative Log-Likelihood (NLL)** quantifies how confidently a model assigns probability to the correct class; a lower NLL means more reliable probability estimates. Additionally, **Expected Calibration Error (ECE)** evaluates whether predicted probabilities match observed frequencies—for example, if the model says “70% confident,” it should be right about 70% of the time on those samples. Formally, we group predictions into bins by confidence and compare average confidence to average accuracy within each bin:

\[
\text{ECE} 
= \sum_{m=1}^M \frac{|B_m|}{N} \, \Bigl|\text{acc}(B_m) - \text{conf}(B_m)\Bigr|,
\]

where \(B_m\) is the set of samples binned by similar confidence, \(\text{acc}(B_m)\) is their average accuracy, and \(\text{conf}(B_m)\) is their average predicted probability. Finally, **F1-Score** (the harmonic mean of precision and recall) helps measure effectiveness in settings with class imbalance.

---

### Conclusion

By running **two mirrored experiments** (SST2 → IMDB and IMDB → SST2), we cover both short-text and long-text training scenarios, revealing whether a model’s in-domain gains transfer to differing input lengths and writing styles. The combination of **ensembling** and **knowledge distillation** often yields more robust predictions and better calibration than a single model alone, while still enabling efficient inference through the distilled student. By comparing in-domain and OOD performance on Accuracy, NLL, ECE, and F1-Score, we gain a comprehensive view of each model’s strengths and weaknesses, ultimately guiding improvements in both architecture design and generalization capabilities.