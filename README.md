# Tutorial: Face Swapping with Google Colab
Face swapping is a popular computer vision technique that allows you to replace one person's face in an image or video with another person's face. In this tutorial, I will walk you through the steps to perform face swapping using Google Colab, a cloud-based Jupyter notebook environment that provides free access to GPUs.

Note: Before you begin, make sure you have a Google account, as you'll need it to access Google Colab.

## Step 1: Setting up Google Colab
Open your web browser and go to Google Colab.

Sign in with your Google account if you're not already signed in.

Click on "New Notebook" to create a new Colab notebook.

## Step 2: Importing Required Libraries
In this tutorial, we'll be using the **dlib** library for face detection and the face_recognition library for face swapping. To install these libraries, add the following code to your Colab notebook cell and run it:
```
!pip install dlib
!pip install face_recognition
```

## Step 3: Uploading Images
You'll need two images for face swapping: one source image (the face you want to replace) and one target image (the face you want to superimpose). You can upload these images to Google Colab by running the following code in a new cell or by clicking the upload button:
```
from google.colab import files
uploaded = files.upload()
```
This code will prompt you to select the source image file from your local computer and upload it to your Colab environment. Repeat the process to upload the target image as well.

## Step 4: Performing Face Swapping
Now that you have your source and target images uploaded, you can perform face swapping. Add the following code to a new Colab cell:
````
import face_recognition
import cv2
import numpy as np

# Load the source and target images
source_image = face_recognition.load_image_file("source.jpg")
target_image = face_recognition.load_image_file("target.jpg")

# Find the face landmarks in both images
source_face_landmarks = face_recognition.face_landmarks(source_image)
target_face_landmarks = face_recognition.face_landmarks(target_image)

# Extract the face landmarks from the source and target images
source_face_landmark_points = source_face_landmarks[0]
target_face_landmark_points = target_face_landmarks[0]

# Create a mask from the target face landmarks
mask = np.zeros_like(target_image)
cv2.fillPoly(mask, [np.array(target_face_landmark_points.values())], (255, 255, 255))

# Warp the source face to match the target face
warped_source_face = cv2.warpAffine(source_image, cv2.estimateAffinePartial2D(np.array(source_face_landmark_points.values()), np.array(target_face_landmark_points.values()))[0], (target_image.shape[1], target_image.shape[0]))

# Combine the warped source face and target face using the mask
output_image = cv2.seamlessClone(warped_source_face, target_image, mask, (int(target_image.shape[1] / 2), int(target_image.shape[0] / 2)), cv2.NORMAL_CLONE)

# Display the swapped face
cv2.imshow("Face Swapped", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
``````
Make sure to replace "source.jpg" and "target.jpg" with the filenames of your uploaded source and target images. Alternatively, you can click on the play button under the **Deepfake** cell.

This code will perform the face swapping operation and display the result in a new window.

## Step 5: Download the Result
If you are satisfied with the face swap, you can download the resulting image by running the following code in a new cell:
```
cv2.imwrite("face_swapped.jpg", output_image)
files.download("face_swapped.jpg")
```

This code will save the swapped image as "face_swapped.jpg" and download it to your local computer.

Congratulations! You have successfully performed face swapping using Google Colab.

## Caution:
Remember to respect privacy and obtain proper permissions when working with images of individuals, especially if you plan to share the swapped images publicly. Face swapping should be used responsibly and ethically.

# Links:
[Google Colab NoteBook](https://colab.research.google.com/drive/1NG9AoH3QDtC7h97z1Yodmn_CiiGh8Y1T?usp=sharing "Colab")

[Video Tutorial1](https://www.loom.com/share/3974453adf7b4498ba864593f7abd0bd?sid=b8fc5320-ef35-49e4-9bce-c046f734f5d6)

[Video Tutorial2](https://www.loom.com/share/0a8ec0117b284bf39e9046781b13cc94?sid=47888fd3-1ee7-41da-85e9-5bf7c0b50010)

[![Sample Video1](video_thumbnail.jpg)](https://github.com/Chalo1996/DeepFakeLinuxTutorial/blob/main/swapped%20(1).mp4)

[![Sample Video2](video_thumbnail.jpg)](https://github.com/Chalo1996/DeepFakeLinuxTutorial/blob/main/swapped.mp4)

# Question:
**If you were to start this position what would be the first 5 cool AIs you would show students to play with?**

1. Chatbots:
   GPT-3 or GPT-4: Showcase the power of natural language processing by building a chatbot or a text-based AI assistant using models like GPT-3 or its successors. You can create engaging conversations and even integrate them into websites or messaging platforms.
2. Image Recognition:
   YOLO (You Only Look Once): Teach students about object detection and real-time image recognition using YOLO, a popular deep learning model. They can experiment with detecting objects in images or even in live video streams.
3. Reinforcement Learning:
   OpenAI Gym: Introduce students to reinforcement learning by using OpenAI Gym. They can develop and train AI agents to play classic games like CartPole or more complex environments like Atari games.
4. Generative Models:
   StyleGAN or BigGAN: Explore the fascinating world of generative models by generating realistic images. Students can create art, generate photorealistic faces, or even design their own unique creatures using these models.
5. Autonomous Vehicles:
   Donkey Car: Give students a taste of autonomous vehicles by building and programming a small-scale self-driving car using Donkey Car. This project combines hardware (e.g., Raspberry Pi and a camera) with machine learning for navigation.






# Other LLMs:
*Deepfacelab*

**Note:**

This LLM requires you to install it to your local machine. Unlike google colab, this LLM uses your local computer resources and it relies heavily upon your machine's computational power (CPU, GPU), RAM, Storage and how compatible the CPU and GPU are. They need to be able to support NVIDIA drivers. Also, more configurations are required thus making it unsuitable for begginers.

This Guide directs you on how to install DeepFaceLab for Linux. Please refer to: [DeepFace](https://github.com/iperov/DeepFaceLab "DeepFace") on how to install on other systems.

## Instalation guide:

**System set up:**

1. Ananconda
   [Anaconda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install "Anaconda")
2. FFMpeg
   [FFMpeg Installation Guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu "FFMpeg")
3. Git
   [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git "Git")
4. NVIDIA
   [NVIDIA Driver Installation Quickstart Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html "NVIDIA")

**Install DeepFaceLab**
[DeepFaceLab_Linux Installation Instructions](https://github.com/nagadit/DeepFaceLab_Linux)
