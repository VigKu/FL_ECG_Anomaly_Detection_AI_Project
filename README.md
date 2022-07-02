# Federated Learning in ECG Anomaly Detection

Abstract:

Can there always be a generalized model in the Federated Learning context? Is there a
way to optimize model training in a Federated environment? In the line of answering these
questions, this ambitious project hypothesizes that controlling the freezing of attention layers
independently can help to enhance the training process and learn new data by shifting the
focus of existing features in the domain of ECG. Through this process, a generalized model
can be formed by just controlling the learning of the attention layers to learn new data which
is advantageous in the context of Federated Learning. A pre-trained model (pre-trained on
MIT-BIH data) with SE-Nets as attention networks are first obtained which is then trained on
two other datasets namely PTB and ECG5000. Adaptive Attention Layer Freezing (AALF)
is implemented and applied to optimally train the attention layers by monitoring the mean F1
score and customized coverage metric for each layer. The experiments conducted in this project
have not produced sufficient favourable results, leading to the rejection of the hypothesis. The
results, however, provide insightful suggestions on the capability of shifting attention layers’
focus and on other areas of the FL training process in TFF.



Major contribution of the Dissertation:

1. The project leverages on existing works for improvement that integrates 2 papers of interests.
(a) Paper 1 - Raza et al. ”Designing ECG Monitoring Healthcare System with Federated
Transfer Learning and Explainable AI” [15] : The project improves on the model
architecture from this paper.
(b) Paper 2 - Chen et al. ”Communication-Efficient Federated Learning with Adaptive
Parameter Freezing” [16] : The project improves on the existing adaptive parameter
freezing (APF) technique from this paper.
2. Based on my research, I believe that this project will be the first to focus on adaptive
freezing for attention layers in the model (technique inspired by APF). This technique
will be called as Adaptive Attention Layer Freezing (AALF).
3. The project is unique to a large extent as it explores into distinguishing a global model
and local model for client use-case with the same model architecture with AALF in the
specific medical domain.
4. The project aims to handle anomaly ECG classes in different feature space with AALF
not seen previously when pre-trained while being cost effective.
5. Previous works in FL have been accomplished with frameworks such as FATE and PySyft
(uses PyTorch at the background). Here, the project uses TFF which is still in its initial
state (Version 0) and thus not able to support fully for practical implementation except
simulation. Thus, this paper contributes to the application/research in TFF.
