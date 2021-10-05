# composite-sketches
Master's thesis project of 2021. Composite sketches generator system. 

Advisor: Zalán Bodó, author: László Antal.

Babeș-Bolyai University of Cluj-Napoca, Romania


# Setup and use of the system

- first download the project source to your local computer
- python verison: `3.7.9`
- install the python dependencies listed in `req.txt` file using anaconda, with the following command: `conda create -n <environment-name> --file req.txt`
- to start the gui you should run the `face_generator_sandbox\main.py` file with python (!! but first you will need the GAN weights file from my Google Drive, just download it and put it in the sandbox folder !!)
- some additional files (weight files, csv files) that are too big to store on git, you can get from my Google Drive: https://drive.google.com/drive/folders/1of81pTbueHti5EGwmUkv1kYemcbukSLg?usp=sharing
- in the Google Drive folder you find 4 files:
1. a demo video (Demo_video.mp4)
2. GAN weights file (karras2018iclr-celebahq-1024x1024.pkl), this file is the same as the one you can find at https://github.com/tkarras/progressive_growing_of_gans. Thank you for sharing it ©Tero Karras!
3. pretrained weights file for the feature extractor (ff_stage-2-rn50.pth)
4. training data of the regression model (regression_train_data2.csv)
- you can find additional experimental results and code snippets for the other parts of the project (e.g. training the custom regression model)



Finally, many thanks to Shaobo Guan, our work is based on his idea. You can find his blog article on Transparent Latent-space GAN at: https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255.


Have fun while you trying it! ;)
With any questions, reach me out at my personal email address: `antal.laszlo011@gmail.com`.
