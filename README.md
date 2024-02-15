# CRAS: Curriculum Regularization and Adaptive Semi-Supervised Learning with Noisy Labels

This repo provides the official PyTorch implementation of our CRAS accepted by Applied Science 2024.

> Abstract: This paper addresses the performance degradation of deep neural networks caused by learning with noisy labels.
Recent research on this topic has exploited the memorization effect: networks fit data with clean labels during the early stages of learning and eventually memorize data with noisy labels.
This property allows for the separation of clean and noisy samples from a loss distribution.
In recent years, semi-supervised learning, which divides training data into a set of labeled clean samples and a set of unlabeled noisy samples, has achieved impressive results.
However, this strategy has two significant problems: (1) the accuracy of dividing the data into clean and noisy samples depends strongly on the network’s performance, and (2) if the divided data are biased towards the unlabeled samples, there are few labeled samples, causing the network to overfit to the labels and leading to a poor generalization performance.
To solve these problems, we propose the curriculum regularization and adaptive semi-supervised learning (CRAS) method.
Its key ideas are (1) to train the network with robust regularization techniques as a warm-up before dividing the data, and (2) to control the strength of the regularization using loss weights that adaptively respond to data bias, which varies with each split at each training epoch.
We evaluated the performance of CRAS on benchmark image classification datasets, CIFAR-10 and CIFAR-100, and real-world datasets, mini-WebVision and Clothing1M.
The findings demonstrate that CRAS excels in handling noisy labels, resulting in a superior generalization and robustness to a range of noise rates, compared with the existing method.