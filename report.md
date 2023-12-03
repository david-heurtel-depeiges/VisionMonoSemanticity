# Towards Monosemanticity: Decomposing Vision Models with Dictionary Learning

#### Quick Disclaimer
This is a research project that is still in progress. I am still working to improve the quality of the code and the clarity of the results. I do not have Anthropic's manpower or compute ressources. I will obviously leave some questions unanswered and this should be taken more as a proof of concept than a definitive answer to the question of monosemanticity in vision models. Most notably, I have not been able to examine fully the result of each model in each hyper-parameter sweep. And the metric we suggest is not perfect.

### Abstract

In artificial neural networks, the most basic computing unit, the neuron, can sometimes exhibit a phenomenon called superposition. This is the ability of a neuron to respond to very different features/inputs. If this is thought to increase the representational power of the network by learning over-complete representations and cramming more information in a single neuron, it can also hinders effort to probe said neural networks and hinder interpretability. In this work, we reproduce Anthropic's work on monosemanticity in language models and apply it to vision models. We show that their method is promising beyond its original scope. We also suggest a metric to evaluate the monosemanticity of a learned dictionary feature.

## Introduction

In 2017 in their Distill article [Feature Visualization](https://distill.pub/2017/feature-visualization/), Olah et al. uncovered neurons in InceptionV1 that were responding to wats, foxed and cars. When probing further and visualizing the features learned by said neuron, one could clearly recognize a patchwork of cats head and cars hoods and windshields. This is a clear example of superposition. Authors wrote:
>"Examples like thse suggest that neurons are not necessarily the right semantic unit for understanding neural nets."

In language models trained by Anthropic, they found similar neurons responding to a mixture of code and natural language. Probing superposition further in their series [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/), they suggest that an independent, monosemantic feature is actually a linear combination of neurons. They contend this increases neural network representation power, increases neural network ability to "represent more independent "features" of the data than it has neurons". Experiments on small toy models suggest this phenomenon arises naturally during training. On the flip side, Anthropic seems to have shown superposition is actually one of the main causes of neural network vulnerability to adversarial attacks. Indeed, if semanticly independent features are actually close linear combinations of neurons, a small perturbation in the input can cause a set of activations to "jump" from one mono-semantic feature to another. This is a very interesting result, but it is not the focus of this work. Superposition is also a problem for interpretability. If a neuron is responding to a mixture of features, it is harder to interpret what it is actually doing. This will be the focus of our work on vision models. We will try to implement one of Anthropic's strategy to find monosemantic features. They developed this method for a 1 layer transformer model. We will apply it to InceptionV1. Said method is a weak dictionary learning technique that relies on a sparse 1-layer deep auto-encoder. By having a very large bottleneck, we allow the auto-encoder to learn an overcomplete basis of neuron activations that is monosemantic. Sparsity, enforced via L1-regularization, is key in forcing the auto-encoder learn a monosemantic or few-semantic representation of the set of neuron activations. Indeed, L1-regularization forces the auto-encoder to reconstruct the input using only a few features/a few vectors in the overcomplete basis. Naturally, the features and vectors in question will be tailored to a specific set of neuron activations. In this work, we show that this method is promising beyond its original scope. We also suggest a metric based on word semanticity (of the ImageNet classes) to evaluate visual monosemanticity. As discussed, this metric is not perfect and makes more sense for the last layers of a vision classifier. For earlier layers, automatic feature evaluation might be done using CLIP embeddings. We leave this for future work and remain open to collaboration (see the TODO section in the Appendix).

## 1. Background and Methods

### 1.1 Background

### 1.2 Architecture and Methods

We will work on InceptionV1 activations. More precisely, we sill work on layers 4a (middle layer, one of the first exhibiting superposition) and layer 5b (last layer before classification head). Here is a reminder of the architecture of InceptionV1:
[<img src="/mnt/home/dheurtel/VisionMonoSemanticity/figures/Inceptionv1L.png" />]

When focusing on a specific layer, we will compute activations after concatenation of all the branches. For a batch of shape $(B, C_{in}, H_{in}, W_{in})$, producing a batch of shape $(B, C_{layer}, H_{layer}, W_{layer})$, we then sum averages activations over the spatial dimensions to get a batch of shape $(B, C_{layer})$. These will be the activations we will work with. Obviously we remove spatial information by doing so and learned features without spatial averaging might be more interesting. Notably, one should perhaps examine whether there exist some form of spatial invariance in the learned features or if there is maybe some form of spatial superposition/polysemanticity. A fixed set of activations may not correspond to the same feature in the top or bottom of the image. But we are getting off topic.

#### 1.2.1 Auto-encoder

We then train an auto-encoder on the $(B, C_{layer})$ activations. Let's have a quick reminder of why this might be a good idea. We present here Anthropic's justification. We want to decompose the activation vector $\mathbf{x}$ of size $C_{layer}$ into a linear combination of $K$ features $\mathbf{d}_k$ of size $C_{layer}$:
$$\mathbf{x} \approx b + \sum_{k=1}^K f_k(\mathbf{x})\mathbf{d}_k$$
where $b$ is a bias term and $f_k(\mathbf{x})$ is a scalar function corresponding to the activation of the $k^{th}$ feature. $\mathbf{d}_k$ is the $k^{th}$ feature direction. The $\mathbf{d}_k$ form the overcomplete basis we mentionned above. We impose that they have unit norm. We know need to determine or at least approximate the $f_k(\mathbf{x})$ functions. We do so by training the auto-encoder. The output of the encoder is:
$$f_k(\mathbf{x}) = \mathrm{ReLU}(\mathbf{W}_e(\mathbf{x} - b_d) + b_e)_k$$
where $\mathbf{W}_e$ is the encoder weight matrix, $b_d$ is the decoder bias and $b_e$ is the encoder bias. One might notice that we first remove the decoder bias. $b_d$ will correspond to the $b$ term in the first equation of this paragraph.
The output of the decoder is then:
$$\mathbf{\hat{x}} = \mathbf{W}_d(f_1(\mathbf{x}), ..., f_K(\mathbf{x})) + b_d$$
where $\mathbf{W}_d$ is the decoder weight matrix. We force column of $\mathbf{W}_d$ to have unit norm. This is equivalent to forcing the $\mathbf{d}_k$ to have unit norm. 

This auto-encoder is then trained with the following loss:
$$\mathcal{L} = \mathbb{E}(||\mathbf{x} - \mathbf{\hat{x}}||_2^2 + \lambda ||\mathbf{f}(\mathbf{x})||_1)$$
where $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), ..., f_K(\mathbf{x}))$ and $\lambda$ is a sparsity parameter.

#### 1.2.2 Training

Training of the auto-encoder is challenging. Most notably, some feature die during training. This was known to happen in Anthropic's work. Periodically, they would "revive" dead features by reinitializing their weights. We do the same. During training, we compute the frequency of activation of each feature over the whole dataset. If a feature activation frequency falls bellow a certain threshold, we reinitialize its weights. Unforntunately, this scheme was very agressive and seems to harm significantly the L2 reconstruction loss. For a discussion about this phenomenon, see the Appendix.

(https://distill.pub/2017/feature-visualization/appendix/)

style="height: 100px; width:200px;"

### 1.3 Analyzing the results

#### 1.3.1 Visual examination of features

The first analysis of the result is manual and visual. We look at a learned feature and the images that activate it the most. We then try to interpret the feature. Conversely, we also take a class or a group of classes (like "dogs" in general) and then look at the features that are the most activated by instances of the class. We then verify that images that activate these features in general have some common semantic property with class instances.

#### 1.3.2 Suggestion of a new metric

Visually examining features is very time consuming and subjective. Ideally, one would have an automated way of evaluating the monosemanticity of a feature. In Anthropic's work, they actually used their own LLM, Claude, to evaluate how monosemantic learned language features were (and Claude was in agreement with human assessment). We do not have an equivalent model.

ImageNet classes, although not a perfect description of our world are actually synsets in the WordNet database/graph. This means that we have a graph of relations (hypernyms, hyponyms...) between ImageNet classes and several other synsets in the WordNet database. Using that, we can build a minimum spanning tree, corresponding to ImageNet hierarchy, with each leaf node corresponding to an ImageNet class and each child of a node being a more precise instance of the father node, an hyponym. Finally, we have a semantic distance between two ImageNet classes that is the length of the shortest path between the two classes in the minimum spanning tree. This induces a matrix $D$ of size $(1000, 1000)$ that is symetric and with zero diagonal.

Given a feature $d_k$, we can then compute the average activation of this feature over each class. This gives a vector in $\mathbb{R}^{1000}$:
$$\mathrm{freq}(d_k)_i = \frac{1}{N_i}\sum_{j=1}^{N_i}f_k(\mathbf{x}_{ij})$$
where $N_i$ is the number of images in class $i$ and $f_k(\mathbf{x}_{ij})$ is the activation of feature $d_k$ for image $j$ in class $i$. 

We propose the following metric to evaluate the monosemanticity of a feature $d_k$:

$$\mathrm{mono}(d_k) = \frac{1}{||\mathrm{freq}(d_k)||_2^2}\mathrm{freq}(d_k)^\intercal D \mathrm{freq}(d_k)$$

This measure how monosemantic a feature is with respect to semanticity of ImageNet classes. If a feature is activated by images of a single class, it will have a low score. If it is activated by a group of very close classes like different breeds of dogs, it will have a slightly higher score, etc...

This metric is far from perfect. Most notably for early layers, it is not clear that ImageNet classes are the best semantic description of the world. 

## 2. Initial Results

### 2.1 Monosemanticity of features

#### 2.1.1 Visual examination of features

#### 2.1.2 Feature visualization

#### 2.1.3 Comparison with neuron activations

### 2.2 Sparsity, how sparse are learned features

### 2.3 Metric evaluation

## 4. Discussion and Perspectives

### 4.3 TODOs and open questions

CLIP metric

## Conclusion

## References

## Appendix

### Appendix A: Training Details

#### Appendix A.1: Vision Models used

#### Appendix A.2: Sparsity, Dead Neurons and Neuron reviving

### Appendix B: Code