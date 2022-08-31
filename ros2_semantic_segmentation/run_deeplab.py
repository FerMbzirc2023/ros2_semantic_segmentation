import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


class DeeplabInference():
  def __init__(self, model_path, ros_structure=False):
    self.model = tf.keras.models.load_model(model_path, custom_objects={"tf": tf})
    self.ros_structure = ros_structure
    self.imagenet_normalization = [103.939, 116.779, 123.68]
    self.mask_id_to_color = {0: (0, 0, 0),     
                        1: (255, 255, 255),
                        2: (255, 0, 0),
                        3: (0, 255, 0),
                        4: (0, 0, 255)}   

  def predict(self, img):
    img_process = img.copy()
    img_process[:,:,0] -= self.imagenet_normalization[0]
    img_process[:,:,1] -= self.imagenet_normalization[1]
    img_process[:,:,2] -= self.imagenet_normalization[2]
    img_process = np.expand_dims(img_process, axis=0)

    prediction = self.model.predict(img_process, verbose=0)          # Shape (batch, h, w, channels)
    prediction = np.squeeze(prediction)                              # Shape (h, w, channels)
    prediction = np.argmax(prediction, axis=2)                       # Shape (index_of_class)

    mask = img.copy()
    for i in self.mask_id_to_color:
      mask[prediction==i] = self.mask_id_to_color[i]

    if self.ros_structure:
      return mask
    else:
      self.visualize(img, mask)


  def visualize(self, img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_mask = img.copy()
    cv2.addWeighted(src1=img, alpha=0.5, src2=mask, beta=0.5, gamma=0, dst=img_with_mask)

    fig = plt.figure(figsize = (10,10))
    axs = np.zeros(3, dtype=object)
    gs = fig.add_gridspec(4, 4, wspace=0.5)
    axs[0] = fig.add_subplot(gs[0:2,1:3])
    axs[1] = fig.add_subplot(gs[2:4,0:2])
    axs[2] = fig.add_subplot(gs[2:4,2:4])

    axs[0].imshow(img_with_mask/255)
    axs[0].set_title('Original image with predicted mask')
    axs[1].imshow(img/255)
    axs[1].set_title('Original image')
    axs[2].imshow(mask/255)
    axs[2].set_title('Predicted mask')
    plt.show(block=False)