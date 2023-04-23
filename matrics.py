import pandas as pd

def get_classification_metrics(df, num_classes):
  metrics = []
  for c in range(num_classes):
      true_positives = ((df['target'] == c) & (df['output'] == c)).sum()
      true_negatives = ((df['target'] != c) & (df['output'] != c)).sum()
      false_positives = ((df['target'] != c) & (df['output'] == c)).sum()
      false_negatives = ((df['target'] == c) & (df['output'] != c)).sum()

      accuracy = (true_positives + true_negatives) / len(df)
      precision = true_positives / (true_positives + false_positives+1e-6)
      recall = true_positives / (true_positives + false_negatives+1e-6)

      metrics.append((accuracy, precision, recall))

  return metrics