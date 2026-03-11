"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees," 
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
Last updated Date: Feburuary 15th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------
main_pategan_experiment.py
- Main function for PATEGAN framework
(1) pategan_main: main function for PATEGAN
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utils import supervised_model_training
from pate_gan import pategan


REPO_ROOT = Path(__file__).resolve().parents[1]
DIABETIC_DATA_PATH = REPO_ROOT / "thesis" / "data" / "diabetic_data_preprocessed_sdv.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "pategan"

#%% 
def pategan_main (args):
  """PATEGAN Main function.
  
  Args:
    data_no: number of generated data
    data_dim: number of data dimensions
    noise_rate: noise ratio on data
    iterations: number of iterations for handling initialization randomness
    n_s: the number of student training iterations
    batch_size: the number of batch size for training student and generator
    k: the number of teachers
    epsilon, delta: Differential privacy parameters
    lamda: noise size
    
  Returns:
    - results: performances of Original and Synthetic performances
    - train_data: original data
    - synth_train_data: synthetically generated data
  """
  
  # Supervised model types
  models = ['logisticregression','randomforest', 'gaussiannb','bernoullinb',
            'svmlin', 'Extra Trees','LDA', 'AdaBoost','Bagging','gbm']
  
  # Load diabetic dataset (already scaled to [0, 1] by sdv_converter)
  data_df = pd.read_csv(DIABETIC_DATA_PATH)
  data = data_df.to_numpy()

  # data_no controls how many rows to use from the dataset
  use_rows = min(args.data_no, data.shape[0])
  data = data[:use_rows]

  # Train/test split
  np.random.seed(42)
  train_ratio = 0.8
  train_mask = np.random.rand(data.shape[0]) < train_ratio
  train_data, test_data = data[train_mask], data[~train_mask]
  data_dim = data.shape[1]
  col_names = data_df.columns
  
  # Define outputs
  results = np.zeros([len(models), 4])
  
  # Define PATEGAN parameters
  parameters = {'n_s': args.n_s, 'batch_size': args.batch_size, 'k': args.k, 
                'epsilon': args.epsilon, 'delta': args.delta, 
                'lamda': args.lamda}
  
  # Generate synthetic training data
  best_perf = 0.0
  
  for it in range(args.iterations):
    print('Iteration',it)
    synth_train_data_temp = pategan(train_data, parameters)
    temp_perf, _ = supervised_model_training(
        synth_train_data_temp[:, :(data_dim-1)], 
        np.round(synth_train_data_temp[:, (data_dim-1)]),
        train_data[:, :(data_dim-1)], 
        np.round(train_data[:, (data_dim-1)]),
        'logisticregression')
    
    # Select best synthetic data
    if temp_perf > best_perf:
      best_perf = temp_perf.copy()
      synth_train_data = synth_train_data_temp.copy()
      
    print('Iteration: ' + str(it+1))
    print('Best-Perf:' + str(best_perf))
  
  # Train supervised models
  for model_index in range(len(models)):
    model_name = models[model_index]
    
    # Using original data
    results[model_index, 0], results[model_index, 2] = (
        supervised_model_training(train_data[:, :(data_dim-1)], 
                                  np.round(train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))
        
    # Using synthetic data
    results[model_index, 1], results[model_index, 3] = (
        supervised_model_training(synth_train_data[:, :(data_dim-1)], 
                                  np.round(synth_train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))

    
    
  # Print the results for each iteration
  results = pd.DataFrame(np.round(results, 4), 
                         columns=['AUC-Original', 'AUC-Synthetic', 
                                  'APR-Original', 'APR-Synthetic'])
  print(results)
  print('Averages:')
  print(results.mean(axis=0))
  
  return results, train_data, synth_train_data, col_names

  
#%%  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_no',
      help='number of generated data',
      default=71518,
      type=int)
  parser.add_argument(
      '--iterations',
      help='number of iterations for handling initialization randomness',
      default=50,
      type=int)
  parser.add_argument(
      '--n_s',
      help='the number of student training iterations',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of batch size for training student and generator',
      default=64,
      type=int)
  parser.add_argument(
      '--k',
      help='the number of teachers',
      default=10,
      type=int)
  parser.add_argument(
      '--epsilon',
      help='Differential privacy parameters (epsilon)',
      default=1.0,
      type=float)
  parser.add_argument(
      '--delta',
      help='Differential privacy parameters (delta)',
      default=0.00001,
      type=float)
  parser.add_argument(
      '--lamda',
      help='PATE noise size',
      default=1.0,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  results, ori_data, synth_data, col_names = pategan_main(args)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  synth_path = OUTPUT_DIR / "diabetic_data_pategan_synthetic.csv"
  results_path = OUTPUT_DIR / "diabetic_data_pategan_results.csv"

  pd.DataFrame(synth_data, columns=col_names).to_csv(synth_path, index=False)
  results.to_csv(results_path, index=False)

  print(f"Synthetic data saved to: {synth_path}")
  print(f"Results saved to: {results_path}")
