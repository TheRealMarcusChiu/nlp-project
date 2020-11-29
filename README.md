# PREREQUISITES
- download the spacy `en_core_web_sm`
- download the nltk wordnet
- pip install spacy nltk torch==1.5.0 torchvision numpy
# RUNNING
- cd into masha/ directory
- for both training and testing:
  - python nlp.py
  - in line 416 you can choose to train or load previous trained model
- for Task 2 only
  - python task2.py