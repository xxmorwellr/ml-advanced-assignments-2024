# Report on Assignment 2
## Basic file structure
*wikiart.py* - define class `WikiArtDataset` and the classification model   
*config.json* - for parameter parsing  
*train.py* - load data and train the classification model   
*test.py* - test the classification model   
*wikiart.pth* - save the modelfile    
For Bonus A and Part 1, I followed the previous structure.

*part_2.py* - for Part 2    
*cluster_plot_n_27.png* - clustering result with n_cluster = 27   
*cluster_plot_n_10.png* - clustering result with n_cluster = 10  

*part_3.py* - for Part 3  
*contrast_image_0.png* - reconstructed one image with five different styles  
*contrast_image_1.png* - reconstructed five images with five different styles

*bonus.py* - for Bonus B  
*prompted_style_image.png* - generated image with prompted style  
For these parts, class `WikiArtDataset` in *wikiart.py* are still applicable.

## Bonus A
I began the assignment with the code demonstrated in former lectures. I want to improve the accuarcy by adjusting the model structure and parameters.

I intially tried to add a convolution layer, and increase the number of `out_channels` to *32* (intial value *1*). Additionally, I increased the `drop_out` rate from *0.01* to *0.3*, which could improve the generalization ability of the model. Till then, the best result was **0.036**.

I tried to add more epochs, add one more convolution layer, adjust `kernel_size`...but all efforts didn't work. I looked into the code and found one possible reason for why accuary is so low: The test set contains *26* classes while the training set contains *27* classes (`Action_painting` is absent) . So when we test the model with the same class `WikiArtDataset`, the mapped class index may be wrong. After the fix, the accuracy increased to **0.16**.

## Part 1 - Fix class imbalancy
I extended class `WikiArtDataset(Dataset)` with `calculate_class_weights()` in *wikiart.py* to calculate class weights. 
```bash 
class_weights: tensor([0.0022, 0.0556, 0.0667, 0.0015, 0.0014, 0.0037, 0.0110, 0.0026, 0.0041,
        0.0009, 0.0061, 0.0051, 0.0004, 0.0041, 0.0047, 0.0027, 0.0244, 0.0024,
        0.0125, 0.0041, 0.0011, 0.0006, 0.0027, 0.0009, 0.0015, 0.0263, 0.0051],
       device='cuda:0')
```
From the output, we can see the class imbalancy did exist.

And then I added the weights as a parameter in criterion function in *train.py*. This time accuracy dropped to only 0.08? I tried to optimize the weight algorithm with `min_weight` and `smoothing_index`, and the accuracy still could be improved to **0.16**.

I also tried to incorporate `p_hash` to filter the influences of similar images, but it seemed unhelpful.
For example, in train set we exactly have one artwork with three variants: original, grayscale-changed and horizontal-flipped one. The p_hash values of the first two images should be the same while the flipped one not.


## Part 2 - Autoencode and cluster representations
I created a new train script for this part. You can see class `Autoencoder` in *part_2.py*, that produces compressed representations of each image in the dataset. I designed *3* convolution layers in my model and chose *MSELoss* to train it.  
Compared to *CrossEntropyLoss*, which is more suitable for classification problems, MSE Loss can help the model accurately restore or compress images at the pixel level, especially in image generation tasks.

For visualizing the clustering result, I utilized *PCA* dimensionality-reduced method. I tried default `n_clusters` setting that equals to the number of art types (27) initially, but the clustering effect seemed not good. When I tried `n_cluster` with 10, the visualization output seemed better. 


## Part 3 - Generation/style transfer
I extended the autoencoder with `style_embedding` in encoder module. You can see class `ConditionalAutoencoder` in *part_3.py*.

I tried to reconstructed one same image with five different styles and reconstructed five images with five different styles. From the output, I could say the style embedding exactly affected generated effect. And when I increased the `style_embedding_dim` and training epochs, the reconstructed output could be better.


## Bonus B
In this part, I loaded *Stable Diffusion* model from Hugging Face and use LoRA to fune-tuning attention layers, allowing lightweight adaptation based on the original weights of the model. To let the image data adapt to the pre-trained model, I also added the preprocessing logic including resize operation.  

To simplify the implementation, I just selected 10 items for training. After fine-tuning, I generated one image with prompted Rococo style -- which is consistent with my expectation. It is more natural and creative than our reconstructed output in Part 3.     
However, limited to constraint computing resources, it's hard to say to what extent it "learned" something from our custom data.  

*When training, it could returned a message like "Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed." It relates to `--no-safety-checks` flag.
