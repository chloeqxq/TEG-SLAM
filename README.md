<p align="center">

  <h1 align="center">WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments</h1>
  <p align="center">
    <a href="https://jianhao-zheng.github.io/"><strong>Jianhao Zheng*</strong></a>
    .
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/valentin-bieri-98426b207/?originalSubdomain=ch"><strong>Valentin Bieri</strong></a>
    .
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    ·
    <a href="https://pengsongyou.github.io"><strong>Songyou Peng</strong></a>
    ·
    <a href="https://ir0.github.io/"><strong>Iro Armeni</strong></a>
</p>
<p align="center"> <strong>Computer Vision And Pattern Recognition (CVPR) 2025</strong></p>
  <h3 align="center"><a href="https://arxiv.org/abs/2504.03886">Paper</a> | <a href="https://www.youtube.com/watch?v=xXuolzFvddQ">Video</a> | <a href="https://wildgs-slam.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
Given a monocular video sequence captured in the wild with dynamic distractors,
WildGS-SLAM accurately tracks the camera trajectory and reconstructs a 3D Gaussian map for static elements, effectively removing all dynamic components. 
</p>
<br>



<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#quick-demo">Quick Demo</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

1. First you have to make sure that you clone the repo with the `--recursive` flag.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 
```bash
git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git
cd WildGS-SLAM
```
2. Creating a new conda environment. 
```bash
conda create --name wildgs-slam python=3.10
conda activate wildgs-slam
```
3. Install CUDA 11.8 and torch-related pacakges
```bash
pip install numpy==1.26.3 # do not use numpy >= v2.0.0
conda install --channel "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip3 install -U xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118
```
4. Install setuptools with a specific version. The latest version will cause some errors for steps 5 and 7. You can also skip this step and do step 5 and 7 without edit mode (remove ``-e``), see [issue](https://github.com/GradientSpaces/WildGS-SLAM/issues/50)
```bash
pip install setuptools==78.1.1
```
5. Install the remaining dependencies.
```bash
python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install -e thirdparty/simple-knn/
```
6. Check installation.
```bash
python -c "import torch; import lietorch; import simple_knn; import diff_gaussian_rasterization; print(torch.cuda.is_available())"
```
7. Now install the droid backends and the other requirements
```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```
8. Install MMCV (used by metric depth estimator)
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```
9. Download the pretained models [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it inside the `pretrained` folder.

## Quick Demo
First download and zip the crowd sequence of Wild-SLAM dataset
```bash
bash scripts_downloading/download_demo_data.sh
```
Then, run WildGS-SLAM by the following command:
```
python run.py  ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml
```

If you encounter a CUDA out-of-memory error, a quick fix is to lower the image resolution. For example, add the following lines to your ```configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml``` file:
```bash
cam:
  H_out: 240
  W_out: 400
```

## Run

### Wild-SLAM Mocap Dataset ([🤗 Hugging Face](https://huggingface.co/datasets/gradient-spaces/Wild-SLAM/tree/main/Mocap))
Download the dataset by the following command. Although WildGS-SLAM is a monocular SLAM system, we also provide depth frames for other RGB-D SLAM methods. The following command only downloads the 10 dynamic sequences. However, we also provide some static sequences. Please check the huggingface page to download them if you are interested in testing with these sequences.
```bash
bash scripts_downloading/download_wild_slam_mocap_scene1.sh
bash scripts_downloading/download_wild_slam_mocap_scene2.sh
```
You can run WildGS-SLAM via the following command:
```bash
python run.py  ./configs/Dynamic/Wild_SLAM_Mocap/{config_file} #run a single sequence
bash scripts_run/run_wild_slam_mocap_all.sh #run all dynamic sequences
```

### Wild-SLAM iPhone Dataset ([🤗 Hugging Face](https://huggingface.co/datasets/gradient-spaces/Wild-SLAM/tree/main/iPhone))
Download the dataset by the following command:
```bash
bash scripts_downloading/download_wild_slam_iphone.sh
```

You can run WildGS-SLAM on any of the sequences via the following command:
```bash
python run.py  ./configs/Dynamic/Wild_SLAM_iPhone/{config_file} #run a single sequence
```
The data is collected by an iPhone and no GT camera pose is available. Therefore, it will be no files related to pose evaluation saved.

### Bonn Dynamic Dataset
Download the data as below and the data is saved into the `./Datasets/Bonn` folder. Note that the script only downloads the 8 sequences reported in the paper. To get other sequences, you can download from the [webiste of Bonn Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html).
```bash
bash scripts_downloading/download_bonn.sh
```
You can run WildGS-SLAM via the following command:
```bash
python run.py  ./configs/Dynamic/Bonn/{config_file} #run a single sequence
bash scripts_run/run_bonn_all.sh #run all dynamic sequences
```
We have prepared config files for the 8 sequences. Note that this dataset needs preprocessing the pose. We have implemented that in the dataloader. If you want to test with sequences other than the ones provided, don't forget to specify ```dataset: 'bonn_dynamic'``` in your config file. The easiest way is to inherit from ```bonn_dynamic.yaml```.

### TUM RGB-D (dynamic) Dataset
Download the data (9 dynamic sequences) as below and the data is saved into the `./Datasets/TUM_RGBD` folder. 
```bash
bash scripts_downloading/download_tum.sh
```
The config files for 9 dynamic sequences of this dataset can be found under ```./configs/Dynamic/TUM_RGBD```. You can run WildGS-SLAM as the following:
```bash
python run.py  ./configs/Dynamic/TUM_RGBD/{config_file} #run a single sequence
bash scripts_run/run_tum_dynamic_all.sh #run all dynamic sequences
```

### Your own dataset
1. Organize your image frames in the following structure:
```yaml
- {Path_to_your_data}
  - rgb
    - frame_00000.png
    - frame_00001.png
    - ...
```

2. Set up your config file using the template at: ``./configs/Custom/custom_template.yaml``. 
Modify the path to your input_folder and change the scene name.
Update the intrinsic parameters to match your dataset.

3. Run WildGS-SLAM!
```bash
python run.py {Path_to_your_config}
```


## Evaluation

### Camera poses
The camera trajectories will be automatically evaluated after each run of WildGS-SLAM (if GT pose is provided). Statistics of the results are summarized in ```{save_dir}/traj/metrics_full_traj.txt```. The estimated camera poses are saved in ```{save_dir}/traj/est_poses_full.txt``` following the [TUM format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).

We provide a python script to summarize the RMSE of ATE [cm]:
```bash
python scripts_run/summarize_pose_eval.py
```

### Novel View Synthesis
Only support for Wild-SLAM Mocap dataset.
```bash
python eval_map/nvs_wild_slam.py
```

## Acknowledgement
We adapted some codes from some awesome repositories including [MonoGS](https://github.com/muskie82/MonoGS), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [Splat-SLAM](https://github.com/google-research/Splat-SLAM), [GIORIE-SLAM](https://github.com/zhangganlin/GlORIE-SLAM), [nerf-on-the-go](https://github.com/cvg/nerf-on-the-go) and [Metric3D V2](https://github.com/YvanYin/Metric3D). Thanks for making codes publicly available. 

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{Zheng2025WildGS,
  author={Zheng, Jianhao and Zhu, Zihan and Bieri, Valentin and Pollefeys, Marc and Peng, Songyou and Armeni Iro},
  title     = {WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## Contact
Contact [Jianhao Zheng](mailto:jianhao@stanford.edu) for questions, comments and reporting bugs.
