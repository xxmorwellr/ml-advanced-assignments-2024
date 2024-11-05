# Report on Assignment 1

## How to run the scripts

My scripts are divided into 3 parts:

- *dataloader.py*: generate training, test, and validation samples
- *train.py*: define and train the model
- *eval.py*: evaluate the model

To run *dataloader.py*, type the following command:

```bash
python dataloader.py [-h] language train_dpi train_font_style test_dpi test_font_style
```
By configuring the following parameters, we can generate the training, test, and validation samples flexibly:

- *language*: specify the language of the characters (e.g., 'Thai', 'English', 'both')
- *train_dpi*: specify the dpi of the training samples (e.g., 200, 300, 400 or 'all')
- *train_font_style*: specify the font style of the training samples (e.g., 'normal', 'bold', or 'all')
- *test_dpi*: specify the dpi of the test samples (e.g., 200, 300, 400 or 'all')
- *test_font_style*: specify the font style of the test samples (e.g., 'normal', 'bold', or 'all')

For further implementation, I save the split datasets to 'train_dataset.pth', 'val_dataset.pth', and 'test_dataset.pth'.

To run *train.py*, type the following command:

```bash
python train.py [-h] epochs learning_rate
```
These hypeparameters are optional, I have set default values in my code (*epochs*=3, *learning_rate*=0.001).

To run *eval.py*, type the following command:
```bash
python eval.py
```
Here `test_dataset` is read using the same *batch_size* with `train_dataset` and `val_dataset`.

## Experiment Analysis
Take the first experiment as an example, run the following commands in sequence:
```bash
 python dataloader.py Thai 200 normal 200 normal
 python train.py
 python eval.py
 ```
 And then we can see the evaluation result. 
 
 I summarized the results of 7 experiments using default parameters:
| Training data                         | Testing data                                                                 | Precision | Recall | F1 Score | Accuracy |
|---------------------------------------|------------------------------------------------------------------------------|-----------|--------|-------|----------|
| Thai normal text, 200dpi              | Thai normal text, 200dpi                                                     | 0.9128          |  0.8923      |   0.8877    |     89.23%     |
| Thai normal text, 400dpi              | Thai normal text, 200dpi |      0.9174     |   0.9032     |    0.9023   |   90.32%       |
| Thai normal text, 400 dpi             | Thai bold text, 400dpi                                                       | 0.9043          |    0.8937    | 0.8914      |    89.37%      |
| Thai bold text                        | Thai normal text                                                             | 0.9232          | 0.9160       |   0.9155    |    91.60%      |
| All Thai styles                       | All Thai styles                                                              |    0.9805       |    0.9797    |   0.9798    |    97.97%      |
| Thai and English normal text jointly  | Thai and English normal text jointly.                                        |     0.9802      |   0.9794     |    0.9793   |       97.94%   |
| All Thai and English styles jointly.  | All Thai and English styles jointly.                                         | 0.9820          |   0.9813     |   0.9814    |   98.13%       |

From the results, I can infer that:
- The `precision`, `recall`, and `F1 score` for both *All Thai styles* and *All Thai and English styles jointly* groups are relatively high, close to 0.98. The current model structure performs well with multiple font styles and mixed languages, and has high generalization ability.
- Different resolutions and font styles have some impact on model performance, but these effects are relatively small.
- Improving the model's generalization ability across different resolutions and font styles may be achieved by adding more diverse training data.

## Practical struggles
- Deal with complex directory structure

  In this assignment, we have a hierarchical file structure, which involves necessary information for splitting datasets and training the model. To learn how to parse the file path is essential.
- Global variable not working
  
  I initially wanted to make `batch_size` adjustable as well, but even if I declared it as *global*, cross-file variable calling was tricky.

  In regard to other shared variable like `num_classes`, I solved the issue by saving and reloading `label_mapping`.


