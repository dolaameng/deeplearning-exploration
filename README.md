# Exploration of Different Deep Learning Frameworks

### Why this project
1. summarize usage patterns of deep learning for different tasks such as images, texts, speech, image-text embedding and etc.
2. write tutorial code on how to use them
3. develop wrapper code to make the use of those frameworks integratable in practice

### What we want to cover
1. how to use a specific framework
2. how to pick the model for different data
3. how to integrate deeplearning models created by specific framework with other machine leanring pipes
4. some insights on how to build deep learning models
5. interesting use cases

### Revision
Picking the right tool is hard, especially when the job you are faced up with is both new and challenging. My initial thought was to compile, play with and test different deep learning frameworks in the literature. Soon I found it is almost impossible to do so, due to the rapid growing pace in deep learning and the update of implementation details (e.g., API) avaiable. Several important things I have learned from this project,
- It might be the trick done by data, rather than algorithms. We have all those fancy results recently for image/speech/text processing, and all those results were obtained by training "deep models" with "huge data". A good theory (e.g. CNN) is definitely necessary to achieve those successes. But beyond that, all the details are more or less on how to make the computation faster, more distributed, capable of accomodating more data, easier to use and etc. Some poeple even argued that given big data, you don't even need fancy algorithms, you just need to be able to train on them.
- The choice right framework heavily depends on the task on your mind. For example, Caffe is fast and convienent if you already know what you want to build, and your data have already been in a good format. But applying Caffe on new raw images, e.g., by deploying it as a web  service is still tedious. On the other hand, other frameworks such as deep4j makes deep learning models more friendly with other machine learning models in a single platform - this is very helpful if need to combine different models for different tasks in your application - recognizing models, mixing it with texts, searching and etc. For learning purpose, I really like the philosophy and api behind theanets and keras - both of them represent the gist of deep learning in their api very well.
- It is more important to get familiar with the ideas and tricks of using net models, e.g., what regularization and intialization are better for what layers, how number of layers influence the performance and etc, than depending heavily on any of existing APIs. It is simply because the field is changing very fast and so are the APIs. A very good source to gain this knowlege is the Stanford online course [CS231n](http://cs231n.github.io/)


## 1. Frameworks to Cover - ordered by degree of interests

- [pylearn2](http://deeplearning.net/software/pylearn2/)
- [theanets](http://theanets.readthedocs.org/en/stable/)
- [neurolab](https://pythonhosted.org/neurolab/)
- [sklearn & related projects](http://scikit-learn.org/stable/related_projects.html)
- [chainer](https://github.com/pfnet/chainer)
- [Caffe](http://caffe.berkeleyvision.org/)
- [deeplearning4j](http://deeplearning4j.org/)
- [torch](http://torch.ch/)
- [Theano](http://deeplearning.net/software/theano/)
- [Decaf - replaced by Caffe](https://github.com/UCB-ICSI-Vision-Group/decaf-release)
- [Overfeat](http://cilvr.nyu.edu/doku.php?id=code:start)
- [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/)
- [deepdist](http://deepdist.com/)
- [pybrain](http://pybrain.org/)
- [Lasagne](https://github.com/Lasagne/Lasagne)
- [keras](https://github.com/fchollet/keras)
- [deepy](https://github.com/uaca/deepy)
- [nolearn](https://github.com/dnouri/nolearn)
- [blocks](https://github.com/mila-udem/blocks)

## 2. Online Tutorials

1. Most frameworks have their own tutorials - see websites for details
2. One of the most enjoyable reading on deep learning basics [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
3. Yoshua's book [Deep Learning](http://www.iro.umontreal.ca/~bengioy/dlbook/)
4. [deeplearning.net](http://deeplearning.net/)
5. [UFLDL stanford](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)
6. [Coursera - Neural Networks for Machine Learning](https://www.coursera.org/course/neuralnets)
7. [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
8. [Interesting read about word2vec, thought it is not "DEEP" really](https://levyomer.wordpress.com/2014/09/10/neural-word-embeddings-as-implicit-matrix-factorization/)
9. [DEEP LEARNING: Methods and Applications](http://research.microsoft.com/apps/pubs/default.aspx?id=209355)
10. [chainer documentation](http://docs.chainer.org/en/latest/)

## 3. Resources

1. [pretrained weights for word2vec](??)
2. [pretrained weights for Caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)
3. [quick installation script guide](installation.txt)
4. [discussion on exsiting deep learning frameworks](http://datascience.stackexchange.com/questions/694/best-python-library-for-neural-networks)
5. [discussion on deep models for specific applications](https://news.ycombinator.com/item?id=9283105)
6. [machine learning pathway](http://www.erogol.com/machine-learning-pathway/) - a wonderful compilation of machine learning

## 4. Exploration Notebooks

## TODO
	- Applications to explore (most of them are keras based)
		- https://github.com/farizrahman4u/seq2seq
		- https://github.com/farizrahman4u/qlearning4k
		- https://github.com/matthiasplappert/keras-rl

		- http://ml4a.github.io/guides/

		- https://github.com/kylemcdonald/SmileCNN
		- https://github.com/jocicmarko/ultrasound-nerve-segmentation
		- https://github.com/abbypa/NNProject_DeepMask
		- https://github.com/awentzonline/keras-rtst

		- https://github.com/phreeza/keras-GAN
		- https://github.com/jacobgil/keras-dcgan

		- https://github.com/mokemokechicken/keras_npi
		- https://github.com/codekansas/keras-language-modeling