# paip_aug


- direct augmentation
  - aug_data
    - allocation.py : Create nerve/tumor allocation dictionary
    - traslation.py : Translate a nerve to tumor areas

- training
  - train
    - train.py            : training module
    - confusion_matrix.py : evaluate trained models

- Sample results

  - PNI classification results by DenseNet161 with default augmentation
  <img src="./images/LA_plot.png" width="700"/>
  
    |                 | precision | recall | f1-score |
    | --------------- | --------- |-------- |-------- |
    | Non-Tumor       | 0.97  |0.94  |0.96  |
    | Tumor w/ Nerve  | 0.96  |0.98  |0.97  |
    | Tumor w/o Nerve | 0.86  |0.90  |0.88  |
  
  - PNI classification results by DenseNet161 with default plus our augmentation strategy
  <img src="./images/LA_plot_aug.png" width="700"/>
  
    |                 | precision | recall | f1-score |
    | --------------- | --------- |-------- |-------- |
    | Non-Tumor       | 0.98  |0.95  |0.97  |
    | Tumor w/ Nerve  | 0.95  |0.98  |0.97  |
    | Tumor w/o Nerve | 0.90  |0.89  |0.89  |
