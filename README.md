#English Articles Correction CNN
Articles 'a', 'an', 'the' correction model based on TensorFlow
## Description:
params: 
- conv: strides=[1, 1, 1, 1], padding="VALID"
- pool: strides=[1, 1, 1, 1], padding="VALID"
- dropout

features = matrix of glove vectors representing surrounding words

model - `text_cnn.py`

test results - `/results`

## Evaluation scores:
* target score = 16.35 %
* accuracy (just for info) = 69.10 %
