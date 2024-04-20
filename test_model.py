import tensorflow as tf  
from tensorflow.keras.preprocessing import image  
import numpy as np  
from pathlib import Path  
  
# This block is setting path 
test_images_dir = input("Enter the test_set_path: ")  # The testset path
model_path = input("Enter the model_path: ")  # the model path  
output_file_path = input("Enter the path to save predictions, you need to firstly create txt file for savind data and input its' path : ")  # the txt  path  
  
# load the model
model = tf.keras.models.load_model(model_path)  

class_name=['adidas','converse','nike']
  
# Get the folder name, in order to set labels 
class_dirs = [d for d in Path(test_images_dir).iterdir() if d.is_dir()]  
  
# Make sure the output is saved at spcialized path 
output_dir = Path(output_file_path).parent  
if not output_dir.exists():  
    output_dir.mkdir(parents=True, exist_ok=True)  
  
# Open the txt file
with open(output_file_path, 'w') as f:  
    # browse all images 
    for class_dir in class_dirs:  
        class_name = class_dir.name  # get feature name
        class_images = list(class_dir.glob('*.jpg'))  # load all images, assume they are jpg
  
        # scan all images in this category 
        for image_path in class_images:  
            # load and pre-process images 
            img = image.load_img(str(image_path), target_size=(108, 108))  # adjust image size
            img_tensor = image.img_to_array(img)  
            img_tensor = np.expand_dims(img_tensor, axis=0)  
            img_tensor /= 255.  # normalize 
  
            # predict by model
            predictions = model.predict(img_tensor)  
            predicted_class = np.argmax(predictions[0])  

            ##set the output calss name
            predicted_class_name=class_name[predicted_class]
  
            # get string from path 
            image_parts = Path(image_path).parts[-2:]  
            category_from_path = image_parts[-2]  
            image_name_from_path = image_parts[-1]  
  
            # formattinf the result
            classname=['adidas','converse','nike']  
            output_line = f'Predicted class for {image_name_from_path} in {category_from_path} is: {predicted_class_name}\n'  
            f.write(output_line)  
  
# Make sure the file is saved
print(f"Predictions have been saved to {output_file_path}")