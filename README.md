# bayesnet

Bayes Network Demo

## Disclaimer
This project is a simplistic implementation of Bayes Network for learning purposes. It is compatible with `pandas.DataFrame` but has the limitation that all columns from the dataset must be categorical (casted as integers e.g. `input/cmc_train.csv`).

## Help

```
Usage:
    python main.py --predict
    python main.py -p wifes_age_1 -c husbands_occ_1,sol_4 -s 1000


Options:
    --predict                 Perform predictions on test dataset
    -s --samples=INT          When specified set the number of samples for Likelihood Weighting
    -p --prob=Event           Hypothesis event
    -c --cond=[<Event1>,...]  List of evidencies
    -h --help                 Print this message
```