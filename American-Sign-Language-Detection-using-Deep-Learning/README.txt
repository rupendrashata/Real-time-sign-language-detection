* First download the dataset from the kaggle site then make a file named input and put all the downloaded dataset into the input file 

* `input` folder contains the the original data from the [Kaggle website] as well as the preprocessed images that are used for training.

* `input/preprocessed_image` contains the resized images that are used for training. The total images in the original dataset is 87000. The `input/preprocessed_image` may contain 87000 or a subset of images depending upon the number of images preprocessed. These many images will be used for training.

* `outputs` folder contains the trained model, the loss and accuracy plots, the predicted test images, and the saved webcam feed with the predicted output.

* `src` folder contains the different python files.
  * `preprocess_image.py`: Preprocess the number of images that you want to use for training.
  * `create_csv.py`: Create a CSV file for the preprocessed images mapping the image paths to the labels. All the images are read from disk during training.
  * `cnn_models.py`: Contains the modules of Custom convolutional neural network model to be used during training.
  * `train.py`: Python file to train the CNN model on the dataset.
  * `test.py`: Python file to test on the images provided in `input/asl_alphabet_test/asl_alphabet_test` folder.
  * `cam_test.py`: Python file for real time webcam sign language detection (**The major aim of this project**). 


* Execute all the files in the terminal while being within the `src` folder.

* `preprocess_image.py`: 

  `python preprocess_image.py --num-images 1200`

  `--num_images` is the number of images to preprocess for each category from `A` to `Z`, including `del`, `nothing`, and `space`.

* `create_csv.py`:

  `python create_csv.py`

* `train.py`:

  `python train.py --epochs 10`

* `test.py`:

  `python test.py --img A_test.jpg`

* `cam_test.py`:

  `python cam_test.py `