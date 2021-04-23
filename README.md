[sample_class_images]: ./results/sample_class_images.png
[model_architecture]: ./results/model_architecture.png
[loss]: ./results/plots/mnetonly-loss.png
[val_kappa]: ./results/plots/mnetonly-valkappa.png
[tsne_before]: ./results/plots/feature_vectors_BEFORE.png
[tsne_after]: ./results/plots/feature_vector_AFTER.png
[exp_table]: ./results/experimental_performance_comparison.png
[lit_table]: ./results/literature_comparison.png

# Diabetic Retinopathy Classification Using a MobileNetV2-SVM Hybrid Model

## Diabetic Retinopathy
Diabetes is a chronic disease characterized by the inability
of the body to process or secrete a sufficient amount of insulin.
One extending complication of this is that diabetes patients are
more prone to diabetic retinopathy (DR), a sight-threatening condition.
Sharp rises in blood sugar level cause blood vessels within the retina 
to become damaged, and blood and fluids leak into surrounding tissue.
This leakage prodcues microaneurysms, hemorrhages, hard exudates,
or cotton wool spots, which contribute towards vision loss. DR is the 
leading cause of blindness among long-term diabetes patients;
20 years after the onset of diabetes, almost all patients and
over 60% with type two experience some degree of retinopathy.

There are 4 main stages of DR:
1. Mild Non-Proliferative
    - Characterised by the formation of microanseurysms and the swelling
    of small blood vessels
2. Moderate Non-Proliferative
    - This condition is classified when the blood vessels begin to experience
    blockage.
3. Severe Non-Proliferative
    - As an increased number of blood vessels are obstructed, certain areas
    become deprived of blood supply.
    - These areas begin to show signs of ischemia such as blot hemorrhages, bleeding
    of veins, and intraretinal microvascular abnormalities.
4. Proliferative
    - Vasoproliferative factors produced by the retina trigger the growth of new abnormal
    and fragile blood vessels.
    
Often, overt symptoms of DR are not observed until an advanced stage is reached. As a result,
individuals diagnosed with diabetes are required to perform regular retinal screening
for timely detection and treatment. However, there are several barriers patients may face, such as
specialist scarcity, time and cost concerns, and insufficient resources. Artificial Intelligence (AI)
can be used to assist in medical screening and diagnosis based on image classification through
deep neural networks.

## APTOS 2019 Dataset
Details about the Asia Pacific Tele-Opthalmology Society (APTOS) 2019 dataset used for this project:
- Consists of 3-channel RGB retina images using fundus photography and corresponding clinician rations for the
severity of DR
- Images taken under a variety of imaging conditions and gathered from multiple clinics
    - Variation and noise within the images and labels; images may contain artifacts, be out of focus,
    underexposed, or overexposed
- Images are different sizes: span 400-5184 pixels in width, 289-3456 pixels in height
- 3662 training images & 1928 test images
- 5 classes
    - 0 - normal (1805 instances)
    - 1 - mild (370 instances)
    - 2 - moderate (999 instances)
    - 3 - severe (193 instances)
    - 4 - proliferative (295 instances)
- Sample images from each class:
    - ![sample_class_images]    
    
## Model Architecture & Training
### Base CNN Model Architecture
The idea for this project was to create a DR severity classifier through a CNN while using fewer parameters
than the more complex models found in past literature. To do this, the MobileNetV2 CNN architecture was leveraged
since it had only around 4.2 million parameters, but was proven to be an effective neural network. The specific
architecture is shown below:
![model_architecture]

However, the real motivation of incorporating a CNN like the one above was to make it a feature extractor. The
embedding retrieved from a trained CNN would then be inputted to an SVM for disease classification. Thus, the
last dense layer with a single neuron shown above was eventually removed and replaced with an SVM head.

### Architecture & Training Details
#### Preprocessing, Data Augmentation, Resampling
A 10-fold cross validation scheme was planned to first train the MobileNetV2 model without a 
separate classifier attached at the end. Each image inputted to the model was resized to 224 by 224 pixels as 
this was the default input resolution of the MobileNetV2 architecture. Since there was a noticeable class 
imbalance in the dataset, all 3296 images in each training partition were resampled to contain 700 instances of each class, 
as this number would ensure the total amount of training images would not fall below the original training partition amount. 
The validation partition images were untouched during each fold to ensure proper evaluation 
of model and its generalizability to unseen data distribution. To reduce the probability of overfitting to the inherent 
noise within the data, on-the-fly image data augmentation was applied to each image in each training partition of the 
cross-validation. This was done to ensure that the resampled images would undergo random transformations before being 
shown to the model to help improve generalizability. One such transformation was applying random zooming within 10% of 
the total image size, which was thought to help create additional noise within each sample without obstructing the 
important medical diagnosis details. Another transformation applied was a random rotation between 0 and 360 degrees, 
since a robust model must be rotationally invariant due to the nature of retina image acquisition and analysis. 
Finally, horizontal and vertical flipping were also randomly applied to further help prevent model overfitting.

#### Deep Neural Network Architecture & Training
All layers of the MobileNetV2 model were trained in this experiment. The architecture of the model was relatively the 
same as found in the original, however, a 2-D global average pooling layer was added to replace the final output layer 
of the original model, followed by two dense layers consisting of 256 neurons each. This configuration was then capped 
off with a final output dense layer of one neuron to retrieve predictions from during the training of the MobileNetV2 model. 
A single neuron was used to ensure the model would output a continuous value since the identification of diabetic 
retinopathy severity in each image was being treated as a regression problem. Conversion of this continuous output to 
the nearest class label value ranging from 0 to 4 was then done to obtain the model’s classification prediction. 
Furthermore, since the classification problem was being approached as an ordinal regression here, the loss function was 
set to mean-squared error (MSE). This was chosen since class target ordinality would assume a higher MSE loss for 
predicted labels that are further apart, such as predicting a proliferative case (class 4) as a normal one (class 0). 
The commonly used Adam optimizer with a small learning rate of 0.0001, along with a small batch size of 32, was 
assigned to the model to promote incremental learning, prevent the model from overshooting during loss minimization, 
and prevent exceeding RAM computation limits during training. The model was initialized using the pretrained weights 
from ImageNet, and then trained on the APTOS dataset for 100 epochs, to become more specific to the DR image problem. 
Finally, a custom callback monitoring the QWK score of the model after each epoch was implemented to save the model 
weights that achieved the highest QWK score during training.

#### Multi-class SVM & Training
Following the completion of training the MobileNetV2 model by itself, a form of transfer
learning was then employed. This was done by loading the model with the best weights as defined by producing the 
highest QWK score during training and removing the final output dense prediction layer. The model’s weights were left 
untrained from this point, thereby designating it as a static feature extractor which would receive a 224 by 224-pixel 
image and output an embedding with 256 dimensions. The removed regression output layer was replaced by an SVM to garner 
more flexibility and potentially higher classification performance. To extend the SVM’s classification ability to a 
multi-class problem, the SVM was used in a One Vs. One approach. This approach was chosen since it was less sensitive 
to any label imbalances in the dataset due to utilizing a smaller subset of data for each SVM classifier instance, as 
compared to the One vs. Rest approach. Additionally, all inputs to the SVM were standardized to help speed up 
computations during training and inferencing. The radial basis function kernel and a cost value of 1 were set after 
performing a 5-fold cross-validation grid search. The SVM prediction head was then trained using 10-fold cross-validation 
with the QWK score as the evaluation metric for each fold.

#### Evaluation Metrics & Experimental Model Variations
The quadratic weighted kappa (QWK) is a commonly used metric for prediction problems on diabetic retinopathy
datasets, due to its adequate ability in illustrating true model performance during multi-class and imbalanced 
class problems. Supplementing this metric, the F1-score which measures the harmonic mean between precision and recall 
of a classifier was also employed. This metric is important for disease classification problems as it accounts for true 
positives and false negatives, the latter of which is critical when dealing with patient condition classification. 
The macro-average F1-score was used since it averages the F1-score individually per class. Thus, in an imbalanced 
dataset, the F1-score penalizes models that perform poorly with minority classes. Finally, to evaluate the model 
against its constituent parts, the model’s performance with these metrics were compared with the MobileNetV2 classifier 
model which had its last dense output layer intact, as well as with an independently trained SVM. Due to the naturally 
high dimensionality of the raw image data, the independent SVM utilized principal component analysis (PCA) by retaining 
components that accounted for 90% of variance. To test the effects of data augmentation, an additional MobileNetV2 
classifier model trained without data augmentation was also put up for comparison. Finally, to test the effect of PCA 
combined with the feature extractor, another version of the MobileNetV2-SVM hybrid model which employed PCA holding 90% 
of variance of the CNN extracted embeddings was also trained and compared with.

## Results
MobileNetV2 Model Training Loss:

![loss]

MobileNetV2 Model Validation Kappa:

![val_kappa]

t-SNE Visualization of Feature Vectors Before & After CNN Embeddings:

![tsne_before] ![tsne_after]

MobileNetV2-SVM Hybrid Model Confusion Matrix:

<img alt="" src="/results/plots/mnet-cf-HIGHRES2.png" width="500px" />

Comparison of Experimental Model Performance:

![exp_table]

Comparison of Hybrid Model Performance:

![lit_table]

Notes on Results:
- Normal class best identified: probably due to high occurrence in dataset
- Mild and Severe (Classes 1 & 3) not distinguished as adequately (and lowest frequency)
- Feature vector embeddings are better delineated after using the MobileNetV2 CNN feature extractor
    - Severity classes are still closer together
- Some overfitting still occurs
- Final MobileNetV2-SVM hybrid model comparable (within 10%) to more complex models in literature