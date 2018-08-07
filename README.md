# Taiko-Master

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/331f51c35b59495395e6a81555c75cfb)](https://app.codacy.com/app/howeverforever/Taiko-Master?utm_source=github.com&utm_medium=referral&utm_content=dsmilab/Taiko-Master&utm_campaign=badger)

This is a project teaching a newbie how to perform better in the video game [**Taiko no Tatsujin**](https://en.wikipedia.org/wiki/Taiko_no_Tatsujin).

<br/>

## Flowchart
In general, follow the paper [Motion Primitive-Based Human Activity Recognition Using a Bag-of-Features Approach](https://dl.acm.org/citation.cfm?id=2110433).

![](docs/flowchart.png)

<br/>

## Preview

### Collect Data

Belows are repos for collecting raw data from wearable devices.


1. [beagle_bone_blue_data_acq](https://github.com/taoyilee/beagle_bone_blue_data_acq)

	- [Introduction of beaglebone-blue](docs/144934_data.pdf): The official manual of BBB.

	- [MPU9250A registration](docs/RM-MPU-9250A-00-v1.6.pdf): The register map and description of MPU-9250

2. [USB-Video-Class-Capture](https://github.com/taoyilee/USB-Video-Class-Capture)

![](docs/video_capture_sample.png)

<br/>

### Singal anaimation

Belows two animations are some extraced features with the specific entire play, and we plot vertical color lines to represent real true hit event. 

![](docs/0420-axyz.gif)
![](docs/0420-gxyz.gif)

<br/>

### Event schematic diagram

We can interpret the local event as the following figure.
![](docs/time_series_sense.png)

<br/>

## Experiment
The followings are all about the song <font color='red'>**夢をかなえてドラえもん**</font>.

<br/>

### Classfication

| hit event | hit event type  | origin hit type |
|----------|:-------------:|------:|
| *no* | 0 | 0 |
| *single* | 1 | 1, 2, 3, 4 |
| *stream* | 2 | 5, 6 |

<br/>

### Model

1. [CNN](util/screenshot_model_generator.ipynb): train the score prediction model.

2. [LGBM](util/doraemon_LGBM.ipynb): train the hit type classification model.

<br/>

### Observation

More observation can be checked at the following notebooks.


1. [Score Vis](util/score_visualization.ipynb): show all plays' score distributions.

2. [Simple Analysis](util/doraemon_analysis.ipynb): briefly visualize training error for all drummers

3. [Confusion Matrix](util/cm_test.ipynb): use model to predict other performance.

In addition, [Taiko-Time-Series-Analytics](https://github.com/taoyilee/Taiko-Time-Series-Analytics) is another related repo analyzing this data.

<br/>

## Result

### Confusion Matrix

#### Case 1.

Using <font color='blue'> **drummer 7** </font> 's 3th play to train and testing 1st, 2nd plays, respectively, then we get

![](docs/test_cnfm-7-3to1.png)
![](docs/test_cnfm-7-3to2.png)

with the feature importance visualization 

![](docs/feature_importance-7.png)

#### Case 2.

Using <font color='blue'> **drummer 4** </font> 's 3th play to train and testing 1st, 2nd plays, respectively, then we get

![](docs/test_cnfm-4-3to1.png)
![](docs/test_cnfm-4-3to2.png)

with the feature importance visualization 

![](docs/feature_importance-4.png)

<br/>

### Self-cross test

Use # to train and # to valid and then test #.

![](docs/barplot_all.png)

Resampling causes increasing training error. On the other hand, scaling causes decreasing one, in general.

<br/>

### All-cross test

![](docs/heatmap_yy.png)
![](docs/heatmap_yn.png)
![](docs/heatmap_ny.png)
![](docs/heatmap_nn.png)

Scaling makes decreasing training error. But resampling has no fixed result.
