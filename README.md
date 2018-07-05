This project is a solution of a kaggle contest - [Digit Recognizer](https://www.kaggle.com/c/digit-recognize), which aims at recognizing single hand-writing digit. This solution is based on CNN and transfer-learning, results in accuracy of 99.61% (with transfer-learning),as a score of top 10%, and 99.40% (without transfer-learning). 

I believe this is one of the best solutions without over-fitting or more relative data, such as the [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset and even MNIST dataset. Considering the test data, which comes from MNIST, is public and labeled, I wonder whether the models of 99.9%+ solutions are kind of over-fitting to the MINST dataset.

To avoid over-fitting, I used lots of dropout-layers and batchnorm-layers. Also, I submit my solution only one time for each model.

You can find more in my blog: [[kaggle]Digit Recognizer(acc 99.61%, top 10%)](https://flouritej.github.io/2018/07/05/kaggle-Digit-Recognizer/)