import tensorflow as tf
import cv2


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    return image


wink = load("wink.jpg")
print(wink.shape, wink.dtype)

# Get a Numpy BGR image from a RGB tf.Tensor
image = cv2.cvtColor(wink.numpy(), cv2.COLOR_RGB2BGR)

cv2.imshow("image", image)
cv2.waitKey()