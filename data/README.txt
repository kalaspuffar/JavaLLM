Training Data
=============

This directory holds training data files used by the JavaLLM demos.

shakespeare.txt (Tiny Shakespeare)
-----------------------------------
A ~1MB plain text file containing a concatenation of Shakespeare's works.
This dataset is public domain and commonly used for character-level
language model experiments.

To obtain it, run the provided download script:

  ./download-shakespeare.sh

Or manually:

  curl -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Alternatively, any plain text file can be used for character-level training.
