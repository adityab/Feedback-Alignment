# Feedback-Alignment
Implementations of [Random Feedback Alignment (RFA)](http://www.nature.com/articles/ncomms13276), [DFA](http://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks), and variants.

## Running

```bash
python3 main.py --lr=0.05 --bs=128 --epochs=100 --flow=autodiff
```

### CLI Options:

- `lr` Learning rate
- `bs` Batch size
- `epochs` Number of epochs to train for
- `flow` The gradient flow scheme used. Possible values:
    - `autodiff` Backprop with Tensorflow's built-in reverse mode automatic differention 
    - `rfa` Random feedback alignment
    - `dfa` Direct feedback alignment