# Voight-Kampff Machine (VKM)

Voight-Kampff Machine is an approach to automatically select a rejection threshold for custom gestures. In high activity (HA) continuous data stream the start and end points of gestures are unknown, and standard approaches to segmentation based on low-activity regions result in high false positive rates. VKM, on the other hand, selects tight rejection threshold so as to minimize the number of false positives and false negatives. This means that with only a few training samples per class, the user can get an accurate custom gesture recognizer even when contiuous data is high activity.

This repository contains a [reference Python VKM implementation](https://github.com/ISUE/VKM) with support for full-body Kinect gestures.   

<p align="center"><img src="https://www.eecs.ucf.edu/isuelab/research/vkm/vkm_diagram.png" width="1000"></p>

## Datasets   

A high activity dataset of four device types (Kinect, Mouse, Vive Position, Vive Quaternion) is included with the publication. The dataset will be automatically downloaded and uncompressed the first time you run the `main.py` file. You can also manually download the dataset [here](https://www.eecs.ucf.edu/isuelab/research/vkm/VKM_datasets_kinect_mouse_vive.zip).

## Running the code

Working on Python 3.9.6 ✅   
Windows:
```cmd
$ git clone https://github.com/ISUE/VKM
$ cd VKM\python
$ python -m venv myenv
$ myenv\Scripts\activate
$ pip install numpy
$ python main.py
```
Linux, Mac (conda is the easiest way to support M1)
```zsh
$ git clone https://github.com/ISUE/VKM
$ cd VKM/python
$ conda create -n myenv python=3.9.6 numpy
$ conda activate myenv
$ python main.py
```
## More Details

For the [publication](https://www.eecs.ucf.edu/isuelab/publications/pubs/Taranta2022.pdf), we evaluated VKM as part of a continuous data processing pipeline, to which we refer to as The Dollar General (TDG) [4]. TDG consists of device-agnostic gesture recognition techniques, and its main components are: Machete [2], which proposes regions that might be gestures; Jackknife [1], which classifies the proposed regions; VKM [this work], which rejects input that does not cross the similarity threshold. To learn more about this research and for technical details on the approach, please refer to the following: 

Project [page](https://www.eecs.ucf.edu/isuelab/research/vkm/) at the ISUE Lab website.   

[1] Taranta II, E. M., Samiei, A., Maghoumi, M., Khaloo, P., Pittman, C. R., and LaViola Jr, J. "[Jackknife: A reliable recognizer with few samples and many modalities](https://www.eecs.ucf.edu/isuelab/research/jackknife/jackknife-final.pdf)." Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems. 2017.

[2] Taranta II, E. M., Pittman, C. R., Maghoumi, M., Maslych, M., Moolenaar, Y. M., and Laviola Jr, J. J. "[Machete: Easy, Efficient, and Precise Continuous Custom Gesture Segmentation](https://www.eecs.ucf.edu/isuelab/publications/pubs/Machete-final.pdf)." ACM Transactions on Computer-Human Interaction (TOCHI) 28.1 (2021): 1-46.
                    

[3] Eugene M. Taranta II, Mehran Maghoumi, Corey R. Pittman, and Joseph J. LaViola Jr. "[A Rapid Prototyping Approach to Synthetic Data Generation For Improved 2D Gesture Recognition](https://www.cs.ucf.edu/~jjl/pubs/uist2016-taranta.pdf)." Proceedings of the 29th Annual Symposium on User Interface Software and Technology. ACM, 2016.


[4] Taranta II, E. M., Maslych, M., Ghamandi, R., and Joseph J. LaViola, Jr. "[The Voight-Kampff Machine for Automatic Custom Gesture Rejection Threshold Selection](https://www.eecs.ucf.edu/isuelab/publications/pubs/Taranta2022.pdf)." CHI Conference on Human Factors in Computing Systems. 2022.

## Citation

If you find yourself using VKM or the High Activity dataset, please reference the following paper:

    @inproceedings{taranta2022_VKM,
        author = {Taranta, Eugene Matthew and Maslych, Mykola and Ghamandi, Ryan and LaViola, Joseph},
        title = {The Voight-Kampff Machine for Automatic Custom Gesture Rejection Threshold Selection},
        year = {2022},
        isbn = {9781450391573},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3491102.3502000},
        doi = {10.1145/3491102.3502000},
        booktitle = {CHI Conference on Human Factors in Computing Systems},
        articleno = {556},
        numpages = {15},
        keywords = {rejection, customization, gesture, recognition},
        location = {New Orleans, LA, USA},
        series = {CHI '22}
    }

## Contributions and Bug Reports

Contributions are welcome. Please submit your contributions as pull requests and we will incorporate them. Also, if you find any bugs, please report them via the issue tracker.

## License

VKM can be used freely for academic research purposes. More details are available in our [license file](https://raw.githubusercontent.com/ISUE/VKM/main/LICENSE).