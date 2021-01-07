# LOTVS-DADA
Driver attention prediction is becoming an essential research problem in human-like driving systems. This work makes an attempt to predict the driver attention in driving accident scenarios (DADA). However, challenges tread on the heels of that because of the dynamic traffic scene, intricate and imbalanced accident categories. In this work, we design a semantic context induced attentive fusion network (SCAFNet). We first segment the RGB video frames into the images with different semantic regions (i.e., semantic images), where each region denotes one semantic category of the scene (e.g., road, trees, etc.), and learn the spatio-temporal features of RGB frames and semantic images in two parallel paths simultaneously. Then, the learned features are fused by an attentive fusion network to find the semantic-induced scene variation in driver attention prediction. The contributions are three folds. 1) With the semantic images, we introduce their semantic context features and verify the manifest promotion effect for helping the driver attention prediction, where the semantic context features are modeled by a graph convolution network (GCN) on semantic images; 2) We fuse the semantic context features of semantic images and the features of RGB frames in an attentive strategy, and the fused details are transferred over frames by a convolutional LSTM module to obtain the attention map of each video frame with the consideration of historical scene variation in driving situations; 3) The superiority of the proposed method is evaluated on our previously collected dataset (named as DADA-2000) and two other challenging datasets with state-of-the-art methods. 

The driving attention map is shown in the following figure：[maps](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/maps.png)

The flowachart of the driver attention model of MSAFNet is as follows.![MSAFNet](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/MSAFNet.png)

Aaccident category: The accident category we defined is shown in the figure:![accident category](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/accident%20classification.jpg)


# Benchmark download:
We are worm-hearted to release this benchmark here, and sincerely invite to use and share it. **Our DADA2000 dataset (about 53GB with compresed mode) can be downloaded from** [Here](https://pan.baidu.com/s/1RfNjeW0Rjj6R4N7beSTYrA). (Extraction code: 9pab) **This is the dataset that we re-uploaded after sorting out.**


Note: DADA benchmark can only be utilized for research, and this downloading link provide a half part of DADA-2000. The reminder half part is confidential and retained for a future competition. If you are interested in this work and the benchmark, please cite the work with following bibtex.
```

@inproceedings{conf/itsc/FangYQXWL19,

  author    = {Jianwu Fang and Dingxin Yan and Jiahuan Qiao and Jianru Xue and He Wang and Sen Li},
  
  title     = {{DADA-2000:} Can Driving Accident be Predicted by Driver Attention?}
               Analyzed by {A} Benchmark},
               
  booktitle = {{IEEE} Intelligent Transportation Systems Conference},
  
  pages     = {4303--4309},
  
  year      = {2019},
  
}
and

@ARTICLE{9312486,
  author={J. {Fang} and D. {Yan} and J. {Qiao} and J. {Xue} and H. {Yu}},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={DADA: Driver Attention Prediction in Driving Accident Scenarios}, 
  year={2021},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TITS.2020.3044678}
}
```
