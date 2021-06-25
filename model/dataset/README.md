# Datasets

The training dataset is synthesized using multiple sources.

* Chords and melodies are obtained from [Hooktheory](https://www.hooktheory.com/)
* The lyrics are obtained by scraping Google



To build the training set:

1. Download the Hooktheory dataset from [this](https://github.com/wayne391/lead-sheet-dataset) repo and copy the `event` folder into this directory, named `hooktheory`
2. Run `python prepocessor.py`
3. The dataset will be built into the folder `processed`