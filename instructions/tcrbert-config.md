
| Field                    | Expected Datatype            | Descriptions                                                                                                                                   |
| ------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `input-path`           | `str`                      | The path to the Dataset                                                                                                                        |
| `output-path`          | `str`                      | Location to flush all outputs                                                                                                                  |
| `model-path`           | `str`                      | The path to the Model                                                                                                                          |
| `maa-model`            | `bool`                     | Whether to use the Masked Amino Acid Model, for details, see[TCR-BERT](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1)'s white paper |
| `negative-dir`         | `list[str]`                | Directory Location to Control Data                                                                                                             |
| `positive-dir`         | `list[str]`                | Directory Location to Cancer Patients                                                                                                          |
| `batch-size`           | `int`                      | Amount of TCR sequences inside each patient to pass into the model at once                                                                     |
| `epoch`                | `int`                      | Amount of Epochs to train the model                                                                                                            |
| `lr`                   | `float` or `list[float]` | Learning Rate, or a List of Learning Rates                                                                                                     |
| `change-lr-at`         | `float` or `list[float]` | Epochs to change Learning Rate at.  This should be the same datatype as "lr"                                                                   |
| `train-split`          | `float`                    | The proportion of data to be served as the training data                                                                                       |
| `bag-accummulate-loss` | `int`                      | The amount of patients to incur a step down the gradient                                                                                       |
| `l2-penalty`           | `float`                    | Amount of Weight Decay                                                                                                                         |
