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