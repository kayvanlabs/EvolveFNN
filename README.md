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
      time_series_data,static_data, labels =    generate_simluated_time_series_data(n_samples,n_timestamp,mislabel = None,random_state=42)```
   * models
   * 
2. Models
   * Layers
   * Network
   * Classifier implementation with early stopping
4. MIMIC
   * Refer to https://github.com/YerevaNN/mimic3-benchmarks/ for in-hospital data generation
5. UM
   * UM data is not available
   
   
