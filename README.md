# Are Emergent Abilities in Large Language Models just In-Context Learning?

> **Abstract:** Large language models have exhibited *emergent* abilities, demonstrating exceptional performance across diverse tasks for which they were not explicitly trained, including those that require complex reasoning abilities. The emergence of such abilities carries profound implications for the future direction of research in NLP, especially as the deployment of such models becomes more prevalent. However, one key challenge is that the evaluation of these abilities is often confounded by competencies that arise in models through alternative prompting techniques, such as in-context learning and instruction following, which also emerge as the models are scaled up. In this study, we provide the first comprehensive examination of these emergent abilities while accounting for various potentially biasing factors that can influence the evaluation of models. We conduct rigorous tests on a set of 18 models, encompassing a parameter range from 60 million to 175 billion parameters, across a comprehensive set of 22 tasks. Through an extensive series of over 1,000 experiments, we provide compelling evidence that emergent abilities can primarily be ascribed to in-context learning. We find no evidence for the emergence of reasoning abilities, thus providing valuable insights into the underlying mechanisms driving the observed abilities and thus alleviating safety concerns regarding their use.

Contact persons: Sheng Lu, Irina Bigoulaeva, Harish Tayyar Madabushi

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

## Evaluation scores
See [evaluation_scores](https://github.com/UKPLab/on-emergence/blob/main/evaluation_scores.csv) for the evaluation scores of our experiments.

## Output files
All of the output files associated to our experiments can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3931). We name each file in a *date_time* fashion (e.g., ```results-20230524_123238.json```), which is also the ```run_id``` of an experiment that can be found in [evaluation_scores](https://github.com/UKPLab/on-emergence/blob/main/evaluation_scores.csv).

## Citation
Please use the following citation:

```
@inproceedings{lu-etal-2024-emergent,
    title = "Are Emergent Abilities in Large Language Models just In-Context Learning?",
    author = "Lu, Sheng  and
      Bigoulaeva, Irina  and
      Sachdeva, Rachneet  and
      Tayyar Madabushi, Harish  and
      Gurevych, Iryna",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.279/",
    doi = "10.18653/v1/2024.acl-long.279",
    pages = "5098--5139",
    abstract = "Large language models, comprising billions of parameters and pre-trained on extensive web-scale corpora, have been claimed to acquire certain capabilities without having been specifically trained on them. These capabilities, referred to as {\textquotedblleft}emergent abilities,{\textquotedblright} have been a driving force in discussions regarding the potentials and risks of language models. A key challenge in evaluating emergent abilities is that they are confounded by model competencies that arise through alternative prompting techniques, including in-context learning, which is the ability of models to complete a task based on a few examples. We present a novel theory that explains emergent abilities, taking into account their potential confounding factors, and rigorously substantiate this theory through over 1000 experiments. Our findings suggest that purported emergent abilities are not truly emergent, but result from a combination of in-context learning, model memory, and linguistic knowledge. Our work is a foundational step in explaining language model performance, providing a template for their efficient use and clarifying the paradox of their ability to excel in some instances while faltering in others. Thus, we demonstrate that their capabilities should not be overestimated."
}
```

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
