# EvolveFNN: An interpretable framework for early detection using longitudinal electronic health record data

This repository contains code related to the paper, "EvolveFNN: An interpretable framework for early detection using longitudinal electronic health record data"

@Author: Yufeng Zhang chloezh@umich.edu

## proposed framework
![network](https://github.com/yufengzhang1995/EvolveFNN/blob/main/network.png)

## Required Packages
* torch==1.12.1
* numpy==1.21.6
* pandas==1.4.1
* scikit-learn==1.2.2
* matplotlib==3.5.1
* shap==0.40.0

## Experiments
1. Simulation
   * data generation
     ```n_split = 5
      n_samples = 1000
      n_timestamp = 10
      random_state = 42
      time_series_data,static_data, labels = generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=42)```
   * models usage
  ```category_info = np.zeros([time_series_data.shape[-1]]).astype(np.int32)
static_category_info = np.zeros(static_data.shape[-1]).astype(np.int32)
static_category_info[0] = 2 
classifier = GeneralizedFuzzyEvolveClassifier(
                evolve_type = 'GRU',
                weighted_loss=[1.0,1.0],
                n_visits = 4,
                report_freq=50,
                patience_step=500,
                max_steps=10000,
                learning_rate=0.1,
                batch_size = 64,
                split_method='sample_wise',
                category_info=category_info,
                static_category_info=static_category_info,
                random_state=1234,
                verbose=2,
                min_epsilon=0.9,
                sparse_regu=1e-3,
                corr_regu=1e-4,
    
            )```
    
2. Models:
   * Layers
   * Network
   * Classifier implementation with early stopping
4. MIMIC
   * Refer to https://github.com/YerevaNN/mimic3-benchmarks/ for in-hospital data generation
5. UM
   * UM data is not available
   
   
