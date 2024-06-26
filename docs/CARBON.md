# **Be Sensible about the Carbon Intensivity when Running your Code**

When employing BERTopic or RoBERTa, it is essential to recognize the broader implications of model selection on environmental sustainability. While BERTopic offers a powerful approach to uncovering latent topics within textual data, the choice of underlying transformer model, significantly impacts computational resources and energy consumption.

Especially RoBERTa, known for its superior performance in natural language understanding tasks, **demands substantial computational power during training and inference stages**, thereby contributing to higher carbon emissions. As such, developers leveraging BERTopic with RoBERTa should prioritize optimization strategies to mitigate the environmental impact of their code. This may entail fine-tuning model parameters, employing efficient batch processing techniques, or exploring alternative models with lower carbon intensities.

Furthermore, it is advisable to preserve your embeddings. By doing so, you eliminate the need to rerun the entire analysis each time, reducing energy expenditure and contributing to a more sustainable workflow.
