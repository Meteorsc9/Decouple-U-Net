# Decouple U-Net
PRCV 2022 "Decouple U-Net: A Method for the Segmentation and Counting of Macrophages in Whole Slide Imaging"

The dataset is available at https://mega.nz/folder/13JWzJ7Q#efCXcG-aFv7Eb_rfbSwUaA
The model is available at https://mega.nz/folder/o742RDha#MO0k0fcJN9-ZIcFdUfheZA

Please unzip the dataset before running.
Example: python main.py train --batch_size=12 --learning_rate=0.0001 --save_path=model.pth --model=DecoupleUNet --epoch=180
python main.py test --weight=/pretrained model/weights_decoupleunet.pth --model=DecoupleUNet 

note: torchmetrics<=0.7.3
