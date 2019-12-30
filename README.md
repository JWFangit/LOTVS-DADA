# LOTVS-DADA
Driver attention prediction has recently absorbed increasing attention in traffic scene understanding and is prone to be an essential problem in vision-centered and human-like driving systems. This work, different from other attempts, makes an attempt to predict the driver attention in accidental scenarios containing normal, critical and accidental situations simultaneously. However, challenges tread on the heels of that because of the dynamic traffic scene, intricate and imbalanced accident categories. With the hypothesis that driver attention can provide a selective role of crash-object \footnote{Crash-object in this paper denotes the objects that will occur accidents.} for assisting driving accident detection or prediction, this paper designs a multi-path semantic-guided attentive fusion network (MSAFNet) that learns the spatio-temporal semantic and scene variation in prediction. For fulfilling this, a large-scale benchmark with 2000 video sequences (named as DADA-2000) is contributed with laborious annotation for driver attention (fixation, saccade, focusing time), accident objects/intervals, as well as the accident categories, and superior performance to state-of-the-arts are provided by thorough evaluations. As far as we know, this is the first comprehensive and quantitative study for the human-eye sensing exploration in accidental scenarios. 

The flowachart of the driver attention model of MSAFNet is as follows.




Aaccident category: The accident category we defined is shown in the figure:![accident category](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/accident%20classification.jpg)

Benchmark download: We are worm-hearted to release this benchmark here, and sincerely invite to use and share it. Our DADA2000 dataset (about 53GB with compresed mode) can be downloaded from [Here](https://pan.baidu.com/s/1gt0zzd-ofeVeElSlTQbVmw).

Note: DADA benchmark can only be utilized for research, and this downloading link provide a half part of DADA-2000. The reminder half part is confidential and retained for a future competition. If you are interested in this work and the benchmark, please cite the work with following bibtex.

@inproceedings{conf/itsc/FangYQXWL19,

  author    = {Jianwu Fang and Dingxin Yan and Jiahuan Qiao and Jianru Xue and He Wang and Sen Li},
  
  title     = {{DADA-2000:} Can Driving Accident be Predicted by Driver Attention?}
               Analyzed by {A} Benchmark},
               
  booktitle = {{IEEE} Intelligent Transportation Systems Conference},
  
  pages     = {4303--4309},
  
  year      = {2019},
  
}
and

@article{conf/ieeeits/FangYQXWL19,

  author    = {Jianwu Fang and Dingxin Yan and Jiahuan Qiao and Jianru Xue},
  
  title     = {DADA: A Large-scale Benchmark and Model for Driver Attention Prediction in Accidental Scenarios},
  
 journal={arXiv preprint arXiv:1912.12148},
 
  year      = {2019},
  
}
