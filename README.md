# Exploring Transformer Interpretability

Transformer models are improving at a rapid pace, making it of paramount importance to develop methods to explain, reverse-engineer, and visualize their inner workings. In this work, we study the interpretability of transformer models through a series of experiments divided into two parts:

1. [Visualizing Transformer Attention](#I.-Visualizing-Transformer-Attention)
2. [Exploring Induction Heads in BERT](#II.-Exploring-Induction-Heads-in-BERT)

This report presents the methods and results of an independent research study conducted over the course of January to April 2023 at the [Harvard Insight and Interaction Lab](https://insight.seas.harvard.edu/) under the mentorship of Catherine Yeh and supervision of Professor Martin Wattenberg and Professor Fernanda Viégas. The full write-up of this project can be found [here](Paper.pdf).



## I. Visualizing Transformer Attention

**AttentionViz.** The self-attention mechanism in transformer models plays a critical role in helping the model learn a rich set of relationships between input elements. To assist in our understanding of attention, [Yeh et al.](https://arxiv.org/abs/2305.03210) developed [AttentionViz](http://attentionviz.com/), a tool that enables the visualization of attention patterns at a more global scale. In particular, AttentionViz introduces a technique for jointly visualizing query and key vectors—two of the core components in computing attention—in a shared embedding space. In AttentionViz, every query and key (originally a 64-dimensional vector) is projected to a 2-dimensional embedding space using t-SNE or UMAP. Queries and keys are jointly displayed on the same plot, allowing for the visualization of distinct attention patterns among queries and keys.

**Distance as a proxy for attention.** A critical idea here is that in the AttentionViz visualizations, we
want distance to be an accurate proxy for attention: high-attention query-key pairs should be closer together in the joint embeddings, a relationship depicted in Figure 1b. To optimize for this desired distance-attention
relationship, we can take a look at how attention is computed based on the q (query), k (key), and v (value)
vectors:

$$ \texttt{attention}(q, k, v) = \textrm{softmax}(\frac{qk^T}{\sqrt{d_k}})v$$

## II. Exploring Induction Heads in BERT
