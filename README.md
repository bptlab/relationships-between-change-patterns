# Relationships between Change Patterns in Dynamic Event Attributes

## Introduction
This repository provides the implementation and setup for the evaluation of the paper entitled <b>Relationships between Change Patterns in Dynamic Event Attributes</b>. This repository includes three jupyter notebooks. The first incorporates the event log generation of the ICU related event log. The second applies the change pattern detection, as introduced in [1]. The third identifies the relationships and illustrates the results in a UI. For the relationship identification, we provide a python [package](https://github.com/bptlab/relationships-between-change-patterns/tree/main/package). The results in the paper refer only to the Departments event log.

## Event Logs (Departments and ICU) 

To reproduce the results, one needs access to the [MIMIC-IV](https://mimic.mit.edu/iv/) database, which requires CITI training. Usually, that does not take much more than a day and access is granted within a week. If access is granted, the event log can be retrieved. We implemented an [event log generation tool](https://github.com/bptlab/mimic-log-extraction/tree/main) for MIMIC-IV, which allows to provide a config file as an input, which results in an ready-to-use event log. Use the [config file](https://github.com/bptlab/relationships-between-change-patterns/blob/main/MIMIC_LOG_CONFIG.yml) in this repository to retrieve the ICU event log by executing the following command: ```python extract_log.py --config MIMIC_Config.yml```. Some post-processing is required, which is conducted in [this jupyter notebook](https://github.com/bptlab/relationships-between-change-patterns/blob/main/1_ICU_Log_Preparation.ipynb). After that, the other jupyter notebooks can be executed with the ICU event log.

The hospital department event log can be extracted, as described in [this repository](https://github.com/jcremerius/Change-Detection-in-Dynamic-Event-Attributes).



## Change Pattern and Relationship Detection

After the event logs have been extracted, change patterns and their relationships can be detected. The implementation is not limited to the above-mentioned event logs and can be used with all event logs. It should be noted, that the event logs should include dynamic event attributes to retrieve any results. It is only required, that the event logs are provided as a .csv file and that the mandatory attributes case id, activity, and timestamp are renamed in the [second jupyter notebook](https://github.com/bptlab/relationships-between-change-patterns/blob/main/2_Applying_Change_Detection.ipynb) accordingly. 

When the change patterns have been detected, execute the [last](https://github.com/bptlab/relationships-between-change-patterns/blob/main/3_UI.ipynb) jupyter notebook. It performs the relationship identification and visualizes the results in an UI.

The figure below illustrates the developed tool for the approach presented in the paper. It shwos an example analysis of the ICU log presented in the paper. On top, the change pattern matrix can be configured, such that only the desired relations, attributes, and trace variants are presented. In the middle, the change pattern matrix is visualized, showing change analysis cells. If a significant change pattern was detected, the cell is coloured according to a value increase (red) or decrease (blue) with its respective RBC value. Clicking on one cell highlights the respective cell, which is the second cell from the left in the last row in Fig. 1. After clicking on one cell, all relevant relationships are illustrated in the table on the right. The respective cells in relationship with the selected cell are highlightes by a black rectangle. Below the table, one relationship can be plotted, which is the first one in the example. The buttons at the bottom allow to visualize change patterns with their relationships in the process model. The p-threshold determines, if a cell in the matrix is a significant change pattern and is not used for the correlation. 

![alt text](https://github.com/bptlab/relationships-between-change-patterns/blob/main/Tool.PNG)
|:--:| 
| *Fig. 1 Tool demonstration* |


[1] Cremerius, J., Weske, M.: Change detection in dynamic event attributes. In: Di Ciccio, C., Dijkman, R., del R ́ıo Ortega, A., Rinderle-Ma, S. (eds.) Business Process Management Forum. pp. 157–172. Springer International Publishing, Cham (2022)
