# EnhancingAI_Project

Welcome to the Project **Hybrid Data for Robust Object Detection in Autonomous Driving**: 
In this repository you can find the presentation that was conducted in preparation to the programming work, the lab diary of the project, the programming files for the 2D object detection model and its results. 

You can run the code by cloning the repository: 
'''bash
git clone https://github.com/elihagen/EnhancingAI_Project.git
'''

You might have to install packages: 
'''bash
pip install tensorflow
'''

To train the model from a jupyter notebook: 
'''bash
%run main.py --dataset "kitti"
'''

To train the model from the terminal: 
'''bash
python main.py --dataset "kitti"
'''

Presentation Link: https://docs.google.com/presentation/d/1BE_FuIC-IuJKvj_FhnQa_sEOvbmL5La0mCjZHp-o9AE/edit?usp=sharing

Lab Diary: https://www.overleaf.com/read/ggvrcjvkpbrc#abb5fb
Troubleshooting: 
created vkitti in csv folder -> mismatch of bboxes? How are bboxes organized in bbox.txt? -> some of the bboxes do not match the image. Must revise how the dataset is generated and how we iterate through the folders
