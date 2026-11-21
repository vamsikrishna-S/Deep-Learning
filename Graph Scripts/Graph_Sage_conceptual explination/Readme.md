 # GraphSAGE: A Conceptual and Practical Guide

## 1. What is GraphSAGE?

GraphSAGE (Graph Sample and Aggregate) is a graph representation learning method. It learns a function that creates a vector, or **embedding**, for each node by combining information from that node and its neighborhood.

Unlike methods that learn a separate embedding table for every node in a fixed graph, GraphSAGE learns shared aggregation functions. This makes it **inductive**: it can generate embeddings for previously unseen nodes, provided their features and neighborhood are available.

GraphSAGE is useful for tasks such as:

- Node classification (for example, categorizing users or documents)
- Link prediction (estimating whether two nodes should be connected)
- Node similarity and retrieval
- Graph-level prediction, when node embeddings are pooled for a whole graph

## 2. Graph terminology

A graph is commonly written as \(G=(V,E)\), where \(V\) is the set of nodes and \(E\) is the set of edges. Each node \(v\) may have a feature vector \(x_v\), such as profile attributes or a text-derived representation.

The **neighbors** of \(v\), written \(N(v)\), are nodes connected to it by edges. A node's **k-hop neighborhood** contains nodes reachable by paths of at most \(k\) edges. GraphSAGE uses a limited number of neighborhood-aggregation layers, so information travels only a limited number of hops.

## 3. The core idea: sample and aggregate

For each node, GraphSAGE repeatedly performs two operations:

1. **Sample neighbors.** Select a bounded number of neighbors at each layer. This controls computation on large graphs. Sampling can be random or use a task-specific strategy.
2. **Aggregate information.** Combine the sampled neighbors' representations into a fixed-size vector, then combine that vector with the node's own representation.

At layer zero, the representation is typically the node's input feature:

\[
h_v^{(0)} = x_v.
\]

At layer \(k\), an aggregator collects the previous-layer representations of sampled neighbors:

\[
m_v^{(k)} = \operatorname{AGGREGATE}_k
	\left(\{h_u^{(k-1)} : u \in S_k(v)\}\right),
\]

where \(S_k(v)\) is the sampled neighborhood used at that layer. The node combines its own representation and the aggregate, then applies a learned transformation and nonlinearity:

\[
h_v^{(k)} = \sigma\left(W_k\,\operatorname{CONCAT}
	(h_v^{(k-1)},m_v^{(k)})\right).
\]

Here, \(W_k\) is a trainable weight matrix and \(\sigma\) is an activation function such as ReLU. Implementations may normalize the result, use a different combination rule, or use a different placement of the transformation. After the final layer, \(h_v^{(K)}\) is the node embedding used by a prediction head or downstream task.

The same layer parameters are shared across nodes. Consequently, the model learns how to transform features and neighborhood information rather than memorizing an independent vector for each node.

## 4. Common aggregators

The aggregator determines how a set of neighbor vectors becomes one vector. The original GraphSAGE work explored several choices:

- **Mean aggregator:** Computes an element-wise mean of neighbor representations. It is simple, permutation-invariant, and efficient.
- **Pooling aggregator:** Applies a learnable transformation such as a dense layer and nonlinearity to each neighbor, then takes an element-wise maximum (or another pooling operation). This can capture richer patterns than a plain mean.
- **LSTM aggregator:** Processes neighbors with an LSTM. Since a neighbor set has no natural order, an ordering or random permutation is used; this makes the aggregation less naturally permutation-invariant.

An aggregator should handle variable-size neighborhoods and should not depend on an arbitrary neighbor ordering when the graph is unordered.

## 5. A small example

Imagine a citation graph where papers are nodes and citations are edges. Each paper has a feature vector derived from its title or abstract. To create an embedding for one paper:

1. Start with that paper's feature vector.
2. Sample a limited number of papers it cites or is cited by, according to the graph's edge convention.
3. Aggregate those neighbors' feature representations.
4. Combine the aggregate with the paper's own representation.
5. Repeat for another layer to include information from a wider neighborhood.
6. Pass the final embedding to a classifier to predict the paper's topic.

The learned aggregation rule can also be applied to a new paper, even if it was not present during training, as long as its features and graph connections are available.

## 6. Training and prediction

GraphSAGE's encoder can be trained with a supervised objective or an unsupervised/self-supervised objective.

### Supervised node classification

For labeled training nodes, a classifier maps the final embedding to class scores. Cross-entropy is a common loss for single-label classification. Only training labels should contribute to the training loss; validation and test labels are reserved for evaluation.

### Link prediction

For a candidate pair \((u,v)\), a scoring function uses their embeddings—for example, a dot product or a small neural network—to estimate the likelihood of an edge. Training typically compares observed edges with sampled non-edges. The choice of negative samples and evaluation split matters because it can substantially affect results.

### Unsupervised representation learning

An objective can encourage connected or contextually related nodes to have compatible embeddings while distinguishing them from sampled negatives. The exact objective depends on the application and sampling scheme.

## 7. Why neighbor sampling matters

If every node has many neighbors, expanding the complete multi-hop neighborhood can become expensive. With fanouts \([s_1,s_2,\ldots,s_K]\), a node's computation is limited to a sampled number of neighbors at each layer. A rough upper bound on the number of sampled nodes for one root is related to \(1+s_K+s_Ks_{K-1}+\cdots\), depending on the convention for assigning fanouts to layers.

Sampling improves scalability, but introduces variance: different samples can produce different embeddings. Larger sample sizes often reduce this variance while increasing memory and computation. During evaluation, use a consistent sampling policy where possible.

## 8. Inductive does not mean graph-free

GraphSAGE can embed unseen nodes because it applies learned functions to their features and sampled neighborhoods. It does **not** mean it can infer useful representations with no information. A new node generally needs:

- Features compatible with those used during training
- A neighborhood, or a deliberate strategy for nodes with no neighbors
- The same feature preprocessing and graph semantics used by the model

If node features are absent or uninformative, an inductive model may have limited ability to distinguish nodes. Adding node IDs as unique features can also undermine generalization to genuinely unseen nodes.

## 9. Practical workflow

1. Define what nodes and edges represent, including edge direction and whether edges are weighted.
2. Prepare node features and labels; fit preprocessing only on training data when required to avoid leakage.
3. Split data carefully. For temporal or relational tasks, ensure future information or target edges do not leak into training neighborhoods.
4. Choose the number of layers, aggregator, and per-layer neighbor fanouts.
5. Train the encoder with a task-appropriate loss and optimizer.
6. Tune choices on validation data, then report final metrics on held-out test data.
7. Check performance for isolated nodes, rare classes, and different neighborhood sizes.

## 10. Design choices and limitations

- **Depth:** More layers provide access to farther nodes but increase cost and can mix information from distant or irrelevant parts of the graph. Deep message-passing models may also suffer from over-smoothing or over-squashing.
- **Fanout:** Small fanouts are cheaper but may omit important neighbors; large fanouts cost more.
- **Features:** Feature quality and consistency are central, especially for inductive use.
- **Neighborhood bias:** Uniform sampling may miss rare but important neighbors. Task-aware sampling can help but must be evaluated carefully.
- **Graph changes:** New nodes can often be embedded, but large distribution shifts or changed feature schemas may require retraining or adaptation.
- **Evaluation leakage:** Edges and labels must be split according to the intended deployment scenario. Accidentally exposing test edges during training can inflate results.

## 11. GraphSAGE compared with transductive embeddings

Methods such as a learned embedding lookup assign a trainable vector to each known node. They can work well when the graph is fixed, but a new node has no learned lookup vector. GraphSAGE instead learns shared transformations over features and neighborhoods, allowing it to compute vectors for nodes not seen as training examples. This flexibility makes it particularly useful when graphs grow or nodes arrive over time.

## 12. Summary

GraphSAGE learns node embeddings by repeatedly sampling neighbors, aggregating their representations, and combining the result with each node's own features. Its shared aggregation functions enable inductive inference, while bounded sampling makes large-graph computation more manageable. Successful use depends on suitable features, sensible sampling and model depth, and careful data splitting and evaluation.

## Further reading

- Hamilton, Ying, and Leskovec, [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), NeurIPS 2017.
