# Smart Scissor: Deep Compressing Both The Image and CNN Architecture for Embedded Hardware

 This repository is provided for reproducing the experimental results reported in the paper.

---
## Environment


* Python: 3.8.10
* Pytorch: 1.8.0
* Numpy: 1.20.2
* CUDA: 11.2
* Yacs: 0.1.8
* Pillow: 8.2.0
* Hardware: RTX-3090

---
## Prepare Data

Here we demonstrate how to get the data ready for evaluation.

**For ImageNet-1K:**
1. Download ILSVRC2012 validation set from [here](https://image-net.org/download.php)
2. Extract the images:

   ```
   mkdir val && tar -xvf ILSVRC2012_img_val.tar -C ./val
   ```
   
3. Copy preprocess_val_dataset.sh to the val directory and re-organize the validation images with:

   ```
   bash preprocess_val_dataset.sh
   ```

7. Finally, the dataset folder should look like:

   ```
   -- dataset 
      -- train
      -- val
   ```
   
**For ImageNet-100:**
1. The ImageNet-100 dataset contains 100 randomly selected classes from ImageNet-1K. The class information of ImageNet-100
is listed in ``configs/imagenet-100.json``.
2. Refer to the preprocessing of ImageNet-1K to organize the ImageNet-100 dataset as follows:

   ```
   -- dataset
      -- train
      -- val
   ```
   
---
## Evaluation
We list the performance of different approaches, and provide the corresponding configurations and pretrained weights for
reproducing the results. The pretrained weights are restored on anonymous Google Drive.


**ResNet50 on ImageNet-1K**   

| Approach | #Params (M) | #MACs (B) | Top-1 Acc. (%)  | Configuration file |  Pretrained |
| ------- | --------- | ------- | -------  | ------- | ------- |
| RCC-Baseline            | 25.6     | 4.1    | 76.0       | ``resnet50.yaml``            | [checkpoint](https://drive.google.com/file/d/15A3IuZYKUAsISOFtjue1w3EKnbh34k-2/view?usp=sharing) |
| **SS-DIC (ours)**       | 25.9     | 4.2    | **77.2**   | ``ss-dic_acc_77.2.yaml``     | [checkpoint](https://drive.google.com/file/d/1LKG3r_-QEY-_GVZTiPiu0lgfKvlWLCUu/view?usp=sharing) |
| RCC                     | 25.6     | 3.0    | 74.4       | ``rcc_acc_74.4.yaml``        | [checkpoint](https://drive.google.com/file/d/1i1u05wJDcrod4rHHMdfslIiZbnW4JQdr/view?usp=sharing) |
| SS-DIC                  | 25.9     | 3.1    | 76.3       | ``ss-dic_acc_76.3.yaml``     | [checkpoint](https://drive.google.com/file/d/1jOOIxTCTsRRrqkp2q5frXgGI_B4v3eIe/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 20.3     | 3.2    | **76.8**   | ``ss-dic-cs_acc_76.8.yaml``  | [checkpoint](https://drive.google.com/file/d/17ZYzdjJBlFt8rP1DdOBoj1rib1RzzbAc/view?usp=sharing) |
| RCC                     | 25.6     | 2.5    | 73.4       | ``rcc_acc_73.4.yaml``        | [checkpoint](https://drive.google.com/file/d/1Pu6VVPd-rImQqDs1TzejtMUAtAdVY-Lu/view?usp=sharing) |
| SS-DIC                  | 25.9     | 2.6    | 75.7       | ``ss-dic_acc_75.7.yaml``     | [checkpoint](https://drive.google.com/file/d/1kHNL9HfNiWxDxFDVlJpnl-LHIOYBnlZE/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 15.4     | 2.4    | **76.3**   | ``ss-dic-cs_acc_76.3.yaml``  | [checkpoint](https://drive.google.com/file/d/1g2_t4zCIpcvbT49FjAaaYmjtQgg-zOyV/view?usp=sharing) |
| RCC                     | 25.6     | 1.8    | 72.7       | ``rcc_acc_72.7.yaml``        | [checkpoint](https://drive.google.com/file/d/1JVBSlRTFXGLueu81d4sPNmQ7e2Uq6n3X/view?usp=sharing) |
| SS-DIC                  | 25.9     | 1.9    | 74.9       | ``ss-dic_acc_74.9.yaml``     | [checkpoint](https://drive.google.com/file/d/1jtelV2yJ17kTU5LRzJfHlThpMyApgfHp/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 13.5     | 1.8    | **75.6**   | ``ss-dic-cs_acc_75.6.yaml``  | [checkpoint](https://drive.google.com/file/d/1_hrdGV4ZZzHUAwLtFRc94e8ZuOodIUQ2/view?usp=sharing) |
| RCC                     | 25.6     | 1.1    | 70.0       | ``rcc_acc_70.0.yaml``        | [checkpoint](https://drive.google.com/file/d/1U2pDwV6mL7YDR2zRTzIaPmk_izCSU1qw/view?usp=sharing) |
| SS-DIC                  | 25.9     | 1.2    | 73.1       | ``ss-dic_acc_73.1.yaml``     | [checkpoint](https://drive.google.com/file/d/1Vg3ww30hnfvtlNZ9nX8TLSBC2TelhW_Z/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 11.7     | 1.3    | **74.2**   | ``ss-dic-cs_acc_74.2.yaml``  | [checkpoint](https://drive.google.com/file/d/19Gm_ek6OO7HrQ6PJ5q9Uqh99AjmUb9el/view?usp=sharing) |


**ResNet50 on ImageNet-100**

| Approach | #Params (M) | #MACs (B) | Top-1 Acc. (%)  | Configuration file |  Pretrained |
| ------- | --------- | ------- | -------  | ------- | ------- |
| RCC                     | 23.7     | 4.1    | 81.6       | ``im100_rcc_acc_81.6.yaml``       | [checkpoint](https://drive.google.com/file/d/10LKOsOY2GpeHYi2r74bIEg_o-f36ssvR/view?usp=sharing) |
| SS-DIC                  | 24.0     | 4.2    | 82.5       | ``im100_ss-dic_acc_82.5.yaml``    | [checkpoint](https://drive.google.com/file/d/1Ivg_mB_lMzYEBuz8FldeKU3UQHJ26zI4/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 17.3     | 3.0    | **82.7**   | ``im100_ss-dic-cs_acc_82.7.yaml`` | [checkpoint](https://drive.google.com/file/d/1MI-FwOUT1VhqG3nT-wXYqxY4Wmpp3URC/view?usp=sharing) |
| RCC                     | 23.7     | 3.0    | 80.1       | ``im100_rcc_acc_80.1.yaml``       | [checkpoint](https://drive.google.com/file/d/1eVNHeOh5N6DIJHpPtwX1L7o_ftxk7iNn/view?usp=sharing) |
| SS-DIC                  | 24.0     | 2.6    | 80.8       | ``im100_ss-dic_acc_80.8.yaml``    | [checkpoint](https://drive.google.com/file/d/1lOIL7uxoYVJk_D9SJQXHiNQ7WS_ubHgy/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 14.5     | 2.4    | **81.5**   | ``im100_ss-dic-cs_acc_81.5.yaml`` | [checkpoint](https://drive.google.com/file/d/1BllcE17chOc0mcZ9hvQJBtj_J9d3TFAi/view?usp=sharing) |
| RCC                     | 23.7     | 1.1    | 76.9       | ``im100_rcc_acc_76.9.yaml``       | [checkpoint](https://drive.google.com/file/d/1YzPtzirs4RSpWKNQF34gHVXXXGPbQeQQ/view?usp=sharing) |
| SS-DIC                  | 24.0     | 1.2    | 77.9       | ``im100_ss-dic_acc_77.9.yaml``    | [checkpoint](https://drive.google.com/file/d/1pdlSY9fj_VaXf6a-Vy8sO5qTOX6cikTD/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 7.8      | 1.0    | **79.3**   | ``im100_ss-dic-cs_acc_79.3.yaml`` | [checkpoint](https://drive.google.com/file/d/1DFmafDv8oVLawhGfXWoLnfio_1xyzjJ0/view?usp=sharing) |


**RegNet-X on ImageNet-100**

| Approach | #Params (M) | #MACs (B) | Top-1 Acc. (%)  | Configuration file |  Pretrained |
| ------- | --------- | ------- | -------  | ------- | ------- |
| RCC                     | 8.4     | 1.6    | 84.8       | ``im100_rcc_acc_84.8.yaml``       | [checkpoint](https://drive.google.com/file/d/1Au9QCG0Zp-OZKeOSRH1Rqhpgifa8WXbs/view?usp=sharing) |
| SS-DIC                  | 8.7     | 1.3    | 85.0       | ``im100_ss-dic_acc_85.0.yaml``    | [checkpoint](https://drive.google.com/file/d/1rVBptPhtLTDNWEomgeS8qEqK4ysUTRlF/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 6.3     | 1.3    | **86.1**   | ``im100_ss-dic-cs_acc_86.1.yaml`` | [checkpoint](https://drive.google.com/file/d/1pNaHYC-Y6lqwHF2Vwc0fIYIJMScnwB05/view?usp=sharing) |
| RCC                     | 8.4     | 0.7    | 82.3       | ``im100_rcc_acc_82.3.yaml``       | [checkpoint](https://drive.google.com/file/d/11zVw3hXsrZj4a5ZT19KIFTEKME7iOK2c/view?usp=sharing) |
| SS-DIC                  | 8.7     | 0.8    | 83.8       | ``im100_ss-dic_acc_83.8.yaml``    | [checkpoint](https://drive.google.com/file/d/1Bx38jPWUrFz9ELNz95307VtmpY9Ikkvy/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 3.6     | 0.6    | **84.0**   | ``im100_ss-dic-cs_acc_84.0.yaml`` | [checkpoint](https://drive.google.com/file/d/1R2BMiByGgSCT7s93Wber1uN89t8f-Ksq/view?usp=sharing) |
| RCC                     | 8.4     | 0.4    | 80.0       | ``im100_rcc_acc_80.0.yaml``       | [checkpoint](https://drive.google.com/file/d/1M19xumtLXE3H3dl8sRBBf00QozOu8BdU/view?usp=sharing) |
| SS-DIC                  | 8.7     | 0.5    | 81.4       | ``im100_ss-dic_acc_81.4.yaml``    | [checkpoint](https://drive.google.com/file/d/13eedMqfaVs_hXtGhe9WZEMnhVCAoL6AU/view?usp=sharing) |
| **SS-DIC-CS (ours)**    | 2.5     | 0.4    | **82.8**   | ``im100_ss-dic-cs_acc_82.8.yaml`` | [checkpoint](https://drive.google.com/file/d/1UdlKsZdg4CtBJfsQSlOK-Eg2_YHUe-6-/view?usp=sharing) |

---

### Run Evaluation
1. Download the pretrained weights with the links above.
2. Download the pretrained weights of the foreground predictor from this [url](https://drive.google.com/file/d/1YTPh4fXSuRmIaTW5YjxpZ_MfRyANFZlr/view?usp=sharing)
3. Move the downloaded weights to the specified directory: ``tmp/checkpoints/``
4. Run inference with the following command:

   ```
   python main.py -m eval/crop_eval
                  -c the_corresponding_configuration_file.yaml
   ```

---

### Evaluation examples

**RCC with ResNet50**

To evaluate RCC with ResNet50 on ImageNet-1K, you can use the following command:

   ```
     python main.py -m eval
                    -c configs/rcc_acc_74.4.yaml
   ```

and if everything goes right, you may obtain an output like this:

   ```
   [meters.py: 192]: json_stats: {"_type": "test_epoch", "epoch": "1/100", "max_top1_acc": 74.3820, "max_top5_acc": 92.0120, "mem": 3088, "time_avg": 0.0953, "time_epoch": 11.9148, "top1_acc": 74.3820, "top5_acc": 92.0120}
   ```


**SS-DIC with ResNet50**

To evaluate SS-DIC with ResNet50 on ImageNet-1K, you can use the following command:

   ```
   python main.py -m crop_eval
                  -c configs/ss-dic_acc_76.3.yaml
   ```

and if everything goes right, you may obtain an output like this:
   
   ```
   [meters.py: 192]: json_stats: {"_type": "test_epoch", "epoch": "1/100", "max_top1_acc": 76.2620, "max_top5_acc": 92.7680, "mem": 3147, "time_avg": 0.1044, "time_epoch": 13.0498, "top1_acc": 76.2620, "top5_acc": 92.7680}
   ```


**SS-DIC-CS with ResNet50**

To evaluate SS-DIC-CS with ResNet50 on ImageNet-1K, you can use the following command:

   ```
     python main.py -m crop_eval
                    -c configs/ss-dic-cs_acc_76.8.yaml
   ```
and if everything goes right, you may obtain an output like this:

   ```
   [meters.py: 192]: json_stats: {"_type": "test_epoch", "epoch": "1/100", "max_top1_acc": 76.8100, "max_top5_acc": 93.1580, "mem": 2720, "time_avg": 0.1091, "time_epoch": 13.6340, "top1_acc": 76.8100, "top5_acc": 93.1580}
   ```
