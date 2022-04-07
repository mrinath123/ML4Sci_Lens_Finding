# ML4Sci DEEP_LENSE


## Common Test approach
### Files
1) common_task_train (train the model)
2) common_task_test (Tests the model on test data)
3) common_task_ROC curve with AUC

The Dataset consists of three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure. 

For this task I coded a custom model  which inspired by the idea of Densenet(https://arxiv.org/abs/1608.06993) and Convolutional Block Attention Module(CBAM)(https://arxiv.org/abs/1807.06521).

For the Loss, Cross_Entropy was doing fine, but it was overfiiting. 
So I read about Label Smoothing BCE (from here https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06)
What it is doing is that, it is decreasing the amount of overconfident predictions(very close to 1 or 0). Which maybe responsible for overfitting, the loss function here works as a regularizer.

### Training trick
For the first few epochs(here 3 epochs) CrossEntropy Loss is used to make the model confident.
After that Label_Smoothing loss is used.
After many experimentations I have found this gives the best performing model

### Results
On Test data got micro_average_area = 0.9954 , macro_average_area = 0.995 , AUC for no substructure class  = 0.99715 , AUC for spherical class = 0.9913 , AUC for vortex class = 0.9957


## Gravitational Lens Finding for Dark Matter Substructure approach

### Files
1) G_lensing(Train) (train the model)
2) G_lensing(Test) (Tests the model on test data)
3) G_lensing_ROC curve 

A data set comprising images with and without strong lenses. 

For this task I developed a **Image and tabular data combined** solution. Only using image was giving good results but when I added the meta fetures it was giving the best.

The model uses transfer learning and uses both the image and meta features for final prediction. Here I use the *tf_efficientnet_b2_ns* backbone. Before the final layer of the model, the meta fetures are also go through a Linear layer and both the image's and meta feature's embeddings are concatenated before giving the final prediction.

### Results

The model on test (10% of whole data) got AUC of 0.977.





