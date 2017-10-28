# 2 Neural Transition-Based Dependency Parsing (50 points)

Some thoughts:

In this assignment, train a basic fully connect neuron network to predict the
dependency trainsition (i.e. SHIFT, LEFT-ARC or RIGHT-ARC). It's a typical
classification problem with 3 classes. The input is a vector concatenated from
36 word vectors of selected tokens (e.g., the last word in the stack, first word
in the buffer, dependent of the second-to-last word in the stack if there is
one, etc.). Given the dimension of the word vector is 50, the concatenated
vector is 1800-dimensional (50 x 36).


Answer to dependency table:

| ROOT, parsed, this           | sentence, correctly |                     | SHIFT     |
|------------------------------|---------------------|---------------------|-----------|
| ROOT, parsed, this, sentence | correctly           |                     | SHIFT     |
| ROOT, parsed, sentence       | correctly           | sentence -> this    | LEFT-ARC  |
| ROOT, parsed                 | correctly           | parsed -> sentence  | RIGHT-ARC |
| ROOT, parsed, correctly      |                     |                     | SHIFT     |
| ROOT, parsed                 |                     | parsed -> correctly | RIGHT-ARC |
| ROOT                         |                     | ROOT-parsed         | RIGHT-ARC |

Here is the training log:

```
================================================================================
TRAINING
================================================================================
Epoch 1 out of 10
924/924 [==============================] - 12s - train loss: 0.1574
Evaluating on dev set - dev UAS: 85.40
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 2 out of 10
924/924 [==============================] - 11s - train loss: 0.0837
Evaluating on dev set - dev UAS: 87.36
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 3 out of 10
924/924 [==============================] - 12s - train loss: 0.0698
Evaluating on dev set - dev UAS: 87.92
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 4 out of 10
924/924 [==============================] - 11s - train loss: 0.0602
Evaluating on dev set - dev UAS: 88.14
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 5 out of 10
924/924 [==============================] - 11s - train loss: 0.0523
Evaluating on dev set - dev UAS: 88.37
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 6 out of 10
924/924 [==============================] - 12s - train loss: 0.0454
Evaluating on dev set - dev UAS: 88.22

Epoch 7 out of 10
924/924 [==============================] - 12s - train loss: 0.0390
Evaluating on dev set - dev UAS: 88.42
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 8 out of 10
924/924 [==============================] - 12s - train loss: 0.0334
Evaluating on dev set - dev UAS: 88.15

Epoch 9 out of 10
924/924 [==============================] - 12s - train loss: 0.0279
Evaluating on dev set - dev UAS: 87.98

Epoch 10 out of 10
924/924 [==============================] - 12s - train loss: 0.0233
Evaluating on dev set - dev UAS: 88.10

================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set - test UAS: 88.91
Writing predictions
Done!
```
