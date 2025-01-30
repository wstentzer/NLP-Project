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


**1. Stanford Sentiment Treebank 2 (SST-2)**  
The Stanford Sentiment Treebank 2 (often referred to as SST-2) is a widely used benchmark dataset for binary sentiment classification. It is a subset of the larger Stanford Sentiment Treebank, originally introduced by Socher et al. (2013) in the paper *“Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.”* Below is a concise but detailed overview:

1. **Origin and Motivation**  
   - **Original Source**: The dataset is derived from movie review snippets provided by Rotten Tomatoes.  
   - **Purpose**: SST-2 is specifically focused on binary (positive vs. negative) sentiment classification tasks, making it a standard benchmark for evaluating the performance of natural language processing (NLP) models, particularly in the context of sentence-level sentiment.

2. **Data Composition**  
   - **Structure**: Each instance in SST-2 corresponds to a single sentence, annotated with either a positive or negative sentiment label.  
   - **Annotations**: All labels are provided at the sentence level (binary polarity), unlike the full Stanford Sentiment Treebank which contains fine-grained labels (very negative, negative, neutral, positive, very positive) at multiple parse-tree nodes.  
   - **Data Splits**: The commonly used version in the GLUE Benchmark provides standardized train, development, and test splits. The approximate sizes are:  
     - **Train**: ~67k sentences  
     - **Dev**: ~872 sentences  
     - **Test**: ~1.8k sentences  

3. **Usage and Characteristics**  
   - **Evaluation Metric**: Accuracy is the primary metric for measuring performance.  
   - **Preprocessing**: Tokenization often follows standardized NLP pipelines. Punctuation, casing, and special characters are typically retained to some degree to reflect real-world textual input.  
   - **Benchmark Role**: Owing to its high quality and well-defined train/dev/test splits, SST-2 is commonly used in research for model comparison and ablation studies in sentiment analysis.

4. **Notable Considerations**  
   - **Sentence-Level Focus**: Since sentences are relatively short, nuances of context and discourse-level features are limited compared to full-text reviews.  
   - **High-Quality Annotations**: The dataset is known for its careful curation, but real-world sentiments can be more diverse than strict binary labels allow.  
   - **Imbalance**: While the dataset is relatively balanced, minor distribution differences between positive and negative classes may still exist, which researchers sometimes address through sampling or weighting strategies.  

**References**  
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). *Recursive deep models for semantic compositionality over a sentiment treebank*. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP).

---

**2. IMDB Movie Review Dataset**  
The IMDB dataset, compiled by Maas et al. (2011), is another prominent corpus for sentiment analysis. It is specifically designed for binary sentiment classification of full-length movie reviews. Below is an overview:

1. **Origin and Motivation**  
   - **Original Source**: The data consists of movie reviews from the Internet Movie Database (IMDB).  
   - **Purpose**: By providing full-text reviews—often longer and more detailed than snippets—the dataset aims to evaluate sentiment analysis techniques in a more realistic setting, where context and discourse structures can significantly influence sentiment.

2. **Data Composition**  
   - **Structure**: Each instance is a movie review in English, with considerable variation in length (some reviews are only a few sentences, while others are multiple paragraphs).  
   - **Labels**: Each review is labeled as either “positive” or “negative,” with no neutral or mixed category. Reviews rated ≥7 stars on IMDB are usually considered positive, and those ≤4 stars negative; the intermediate ratings are removed to ensure a clear polarity distinction.  
   - **Size and Splits**:  
     - **Train Set**: 25,000 labeled reviews (balanced between positive and negative).  
     - **Test Set**: 25,000 labeled reviews (also balanced).  
     - **Unlabeled Data**: An additional set of 50,000 reviews is often included for unsupervised or semi-supervised techniques.

3. **Usage and Characteristics**  
   - **Evaluation Metric**: Accuracy is the standard metric. Researchers also examine other metrics such as F1-score, precision, and recall.  
   - **Realistic Text Length**: Reviews can contain multiple sentences or paragraphs, testing a model’s ability to handle long-form text with complex linguistic structures.  
   - **Topic and Style Variance**: Since the dataset is collected from diverse movie genres and reviewers, it exhibits significant variation in writing style, vocabulary, and domain-specific references.

4. **Notable Considerations**  
   - **Data Imbalance Solutions**: Though the labeled sets are balanced, the large volume of unlabeled data may be leveraged to improve performance using semi-supervised learning.  
   - **Overfitting Risk**: The long text format increases the risk of overfitting to training data if models do not employ proper regularization or architecture choices.  
   - **Noise and Subjectivity**: User-generated content may contain spelling errors, slang, and strong subjective expressions that challenge standard NLP pipelines.



Below is a concise description of two popular sentiment analysis datasets, Stanford Sentiment Treebank 2 (SST-2) and the IMDB Movie Review dataset, presented in paragraph form for a research paper.

---

### Stanford Sentiment Treebank 2 (SST-2)

The Stanford Sentiment Treebank 2 (SST-2) is a well-established benchmark for binary sentiment classification, introduced by Socher et al. (2013) as part of the larger Stanford Sentiment Treebank. This subset focuses on classifying sentences as either positive or negative and is derived from movie review snippets provided by Rotten Tomatoes. Unlike the full Stanford Sentiment Treebank, which offers fine-grained sentiment labels (ranging from very negative to very positive) at various nodes of a parse tree, SST-2 simplifies these labels into binary categories at the sentence level. The commonly used version from the GLUE Benchmark includes a training set of roughly 67,000 sentences, a development set of about 872 sentences, and a test set of approximately 1,800 sentences. Researchers typically adopt accuracy as the primary metric to evaluate performance on SST-2. Despite the dataset’s generally balanced distribution of classes, minor imbalances can occur between positive and negative labels. Its high-quality annotations and standardized splits have made SST-2 a central reference point in sentiment analysis studies, facilitating direct comparisons between different model architectures and training regimes.

---

### IMDB Movie Review Dataset

The IMDB Movie Review dataset, compiled by Maas et al. (2011), is another influential corpus for binary sentiment classification. In contrast to SST-2’s short snippets, the IMDB dataset consists of full-length movie reviews obtained from the Internet Movie Database (IMDB). Each review is labeled as positive or negative, with ratings of seven stars or higher treated as positive, and ratings of four stars or lower labeled as negative (intermediate ratings are removed to preserve clear polarity). The standard split provides 25,000 labeled reviews in the training set and 25,000 labeled reviews in the test set, balanced between positive and negative examples. An additional 50,000 unlabeled reviews are often included for unsupervised or semi-supervised learning approaches. Because many reviews are lengthy and diverse in style, topic, and tone, models must capture more complex and context-dependent language patterns than those found in single-sentence data. While accuracy is the primary measure of performance for IMDB, researchers commonly report additional metrics such as precision, recall, or the F1-score to gain deeper insights into their model’s capabilities. The dataset’s breadth and variability make it a robust testbed for evaluating sentiment analysis methods, particularly those aimed at handling extensive textual input.

---

### References

- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). *Recursive deep models for semantic compositionality over a sentiment treebank.* In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.  
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis.* In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL)*.


