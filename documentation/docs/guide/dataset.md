# Dataset

The data used in this project originate from the [NUS Global Streetscapes dataset](https://github.com/ualsg/global-streetscapes), which was published by the National University of Singapore. The dataset contains large-scale street-level imagery with a variety of labels. For this project, these labels were combined into a single large table that includes the complete set of images and metadata.

!!! see also "Notebook"
    For the code and steps used to build the table, see [**Initial Table Generation**](../notebooks/initial_table.ipynb).

## Steps for preparing the subsets

### Step 1

The following table shows how the total count of the chosen cities in the study change when using the `mly_quality_score`. This is quite important to notice, hence, the amount changes quite drastically when using a different thresholds. 

!!! note
    This approach is useful when computation time needs to be reduced and fewer pairs are expected.

| City         | Total  | 50 %  | 60 %  | 70 %  | 80 %  | 90 %  |
|:-----------|------:|-----:|-----:|-----:|-----:|-----:|
| Berlin       | 198184 | 61606 | 59728 | 56517 | 51531 | 41767 |
| Washington   | 197080 | 76859 | 70041 | 60128 | 44313 | 24971 |
| Sydney       | 69227  | 63944 | 61771 | 57759 | 52210 | 41051 |
| Cape Town    | 12639  | 11135 | 10136 | 8764  | 6708  | 4068  |
| Taipei       | 198538 | 171232| 161789| 146595| 122037| 84761 |
| Sao Paulo    | 197964 | 129330| 108546| 78852 | 46080 | 19913 |

Since testing showed, that the `heading` variable is not a 100 % reliable, *mapillary's* `computed heading` was used as well to determine if there is an offset greater than 10 degrees. The following reduction of the table looks the following:

| City       | Total | 50 % | 60 % | 70 % | 80 % | 90 % |
|:-----------|------:|-----:|-----:|-----:|-----:|-----:|
| Berlin       | 45650  | 13035 | 12476 | 11629 | 10287 | 8018  |
| Washington   | 132952 | 51160 | 47204 | 41233 | 30535 | 16644 |
| Sydney       | 15304  | 12849 | 12164 | 11228 | 9924  | 7573  |
| Cape Town    | 9511   | 8276  | 7416  | 6303  | 4618  | 2594  |
| Taipei       | 26417  | 19805 | 18686 | 16351 | 12673 | 8040  |
| Sao Paulo    | 124394 | 79460 | 66792 | 48881 | 27401 | 11107 |

As a result, the difference between the headings and the score were being used to reduce the size of the images.

### Step 2

Since the initial amount of images was still too big, we decided to use the tool by [Danish et al. (2024)](https://arxiv.org/abs/2403.00174) which can be found on [GitHub](https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept). For our usage, the tools were slightly modified.
How to use it, is explained in the See the [Advanced guide](advanced.md). 

## Final dataset

The result of these steps is a cleaned and filtered subset of the Global Streetscapes dataset. In the Berlin example, this produced a smaller but higher quality collection of images that can be used for tasks such as pair identification and perception studies.