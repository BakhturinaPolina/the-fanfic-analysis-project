# The Fanfic Analysis Project

1. `booknlp_processing.py` uses `booknlp` to process the texts, tokenizing them etc.
2. `lemma_extraction.py` filters the resulting dataframes to create three versions of the corpus (all without proper nouns, verbs+nouns only, nouns only), then gets lemma values, then does some residual cleanup
3. `combine_tokens.py` combines the tokens for each text into a single file (i.e. three single files, one for each corpus type)
4. `train_model.py` tries different parameters to find a model with the highest coherence