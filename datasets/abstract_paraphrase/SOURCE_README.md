# General information and usage instructions

This repository contains the following folders:

1. code -- with Python codes needed to tune/train classification models and to test models on previously unseen data;
2. data -- with sample of all English and Serbian data (PhD theses abstracts from the Faculty of Architecture);
3. test -- with additional cmd scripts used to call the training of models or to test already trained models.

To train models on data samples, you can simply call the script test/train_en.cmd or test/train_sr.cmd.
If you want to train models on your data (English or Serbian), you should change the parameters in these scripts to take the target paths into account.

If you want to test trained models, you can customize test/test_trained_model.cmd.
Currently, it is set to run the serialized XGB classifier (as described in the paper) with word unigrams on English data.

Not all serialized models are available in this GitHub repository, as we exported models larger than 100 MB to a Google Drive to avoid an LFS setup:
https://drive.google.com/drive/folders/1HvhTJ7HLhW9YzLJBFZir-30wXP4viyOg?usp=sharing
To access these models, simply download them from Google Drive and unzip them to the appropriate path in the test/models folder where other smaller models are already located.
