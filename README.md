# Grooming-JTBD
Cascading models aiming to generate DGE labels or 'unknown' (OOD) labels for given texts.
Input text goes into a BERT binary classification model to see if it is related to grooming problems/painpoints.
For those related texts, they then need to pass through a finetuned GPT-3 Davinci model to get corresponding JTBD (DGE) labels.
