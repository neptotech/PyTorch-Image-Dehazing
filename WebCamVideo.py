# import the opencv library
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

import net

# define a video capture object
vid = cv2.VideoCapture(0)


def dehaze_image(image):
    data_hazy = image
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()

    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cpu().unsqueeze(0)

    dehaze_net = net.dehaze_net().cpu()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth', map_location=torch.device('cpu')))

    clean_image = dehaze_net(data_hazy)
    return clean_image
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    print(dehaze_image(frame).permute(1,2,0))
    # Display the resulting frame
    plt.imshow(transforms.ToPILImage()(dehaze_image(frame)), interpolation="bicubic")
    plt.show()
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()