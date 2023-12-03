# CLIPForImageClassification

While standard image models train an image feature
extractor and a linear classifier to predict the probability distribution over class labels, [CLIP](https://arxiv.org/abs/2103.00020) jointly trains an
image encoder and a text encoder to predict the correct pairings of a batch of
(image, text) training examples. At test time the learned text encoder synthe-
sizes a zero-shot linear classifier by embedding the names or descriptions of the
target dataset’s classes.

## Implementation of OpenAI CLIP model directly for Image Classification task.

[CLIPForImageClassification Model Card](https://huggingface.co/Andron00e/CLIPForImageClassification-v1),  [OpenAI CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md)

## Useful links
[OpenAI model on hub](https://huggingface.co/openai/clip-vit-large-patch14)

[Our CLIPForImageClassification model on hub](https://huggingface.co/Andron00e/CLIPForImageClassification-v1)

[Google ViTForImageClassification model on hub](https://huggingface.co/google/vit-base-patch16-224)

[Our ViTForImageClassification model on hub](https://huggingface.co/Andron00e/ViTForImageClassification)

[Colab demo](https://colab.research.google.com/drive/1ZWDCYou8E3nuFJPTwXgSidhGpP134R6a#scrollTo=PVDHw8j5-78H)

# Citation

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```    
```bibtex
@misc{dosovitskiy2021image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
