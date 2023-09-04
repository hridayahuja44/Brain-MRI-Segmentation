# Brain MRI Segmentation Model
We've developed a model that's designed to perform segmentation on brain MRI images. This means that it's capable of identifying and highlighting specific regions of interest within these images. The architecture we chose for this task is called U-Net, which is the ideal choice for medical image segmentation tasks. When you provide an MRI image as input, the U-Net model processes it through a series of layers that progressively learn to recognize different features within the image. 

## Input Image to the Model
![Output image](https://github.com/hridayahuja44/Brain-MRI-Segmentation/blob/main/Testing%20Image.png)

## Output Masked Image
![Output image](https://github.com/hridayahuja44/Brain-MRI-Segmentation/blob/main/Testing%20Image%20Mask.png)

Once the model processes the input image, it generates an output mask that highlights the segmented regions. The output mask assigns a value to each pixel, indicating its class or category. This way, you can clearly see which parts of the brain the model has identified as belonging to the region of interest.
- Binary Accuracy: 0.9108
- Intersection Over Union: 0.0492
- dice Coefficients: 0.0897

## Accuracy Graph for the Segmentation Model
![Output image](https://github.com/hridayahuja44/Brain-MRI-Segmentation/blob/main/Accuracy%20Graph%20For%20the%20Model.png)

The U-Net architecture has proven to be quite effective for tasks like medical image segmentation, where precise identification of specific structures is crucial. It's capable of producing accurate segmentation results, which can be extremely valuable for medical professionals in diagnosing and treating various brain conditions. In a nutshell, my brain MRI Segmentation model based on the U-Net architecture takes MRI images as input, processes them through a specialized neural network, and produces a masked output that highlights the segmented regions of interest within the brain images.
  
