import sys

import numpy as np
from sklearn.metrics import confusion_matrix

if len(sys.argv) != 3:
    print("Usage: python eval_confusion_matrix.py eval_output_file qual_threshold")
    sys.exit(1)

eval_output = sys.argv[1]
qual_threshold = float(sys.argv[2])

pred_array = []
truth_array = []
with open(eval_output, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        quality = float(parts[2])
        truth, pred = map(int, parts[3].split(','))
        if quality >= qual_threshold:
            pred_array.append(pred)
            truth_array.append(truth)

pred_array = np.array(pred_array)
truth_array = np.array(truth_array)
cm = confusion_matrix(truth_array, pred_array)
print(cm)
