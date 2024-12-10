# LOTVS-DADA
Driver attention prediction is becoming an essential research problem in human-like driving systems. This work makes an attempt to predict the driver attention in driving accident scenarios (DADA). However, challenges tread on the heels of that because of the dynamic traffic scene, intricate and imbalanced accident categories. In this work, we design a semantic context induced attentive fusion network (SCAFNet). We first segment the RGB video frames into the images with different semantic regions (i.e., semantic images), where each region denotes one semantic category of the scene (e.g., road, trees, etc.), and learn the spatio-temporal features of RGB frames and semantic images in two parallel paths simultaneously. Then, the learned features are fused by an attentive fusion network to find the semantic-induced scene variation in driver attention prediction. The contributions are three folds. 1) With the semantic images, we introduce their semantic context features and verify the manifest promotion effect for helping the driver attention prediction, where the semantic context features are modeled by a graph convolution network (GCN) on semantic images; 2) We fuse the semantic context features of semantic images and the features of RGB frames in an attentive strategy, and the fused details are transferred over frames by a convolutional LSTM module to obtain the attention map of each video frame with the consideration of historical scene variation in driving situations; 3) The superiority of the proposed method is evaluated on our previously collected dataset (named as DADA-2000) and two other challenging datasets with state-of-the-art methods. 

The driving attention map is shown in the following figure：![maps](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/maps.png)

The flowachart of the driver attention model of MSAFNet is as follows.![MSAFNet](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/MSAF_Net.png)

Aaccident category: The accident category we defined is shown in the figure:![accident category](https://github.com/JWFangit/LOTVS-DADA/blob/master/DADA_accident_categories/accident%20classification.jpg)


# Benchmark download:
We are worm-hearted to release this benchmark here, and sincerely invite to use and share it. **Our DADA2000 dataset (about 53GB with compressed mode) can be downloaded from** [Here](https://pan.baidu.com/s/1RfNjeW0Rjj6R4N7beSTYrA). (Extraction code: 9pab) **This is the dataset that we re-uploaded after sorting out.** Then you can use python/extract_images.py to decompress the datasets.

Note: The above released part is the used training and testing set in our work publicated in [1]. Recently, we have released **FULL benchmark (about 116G with compressed mode) and can be downloaded** [here](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz) (Extraction code: ahyz) .

Newa: we have released **FULL benchmark can be downloaded** [Google Drive](https://pan.baidu.com/s/1oxoQKYIaNCkLCxVCrOwgHw?pwd=ahyz).

If you are interested in this work and the benchmark, please cite the work with following bibtex.

```
[1] Jianwu Fang, Dingxin Yan, Jiahuan Qiao, Jianru Xue, Hongkai Yu:
DADA: Driver Attention Prediction in Driving Accident Scenarios. 
IEEE Trans. Intell. Transp. Syst. 23(6): 4959-4971 (2022)
```
