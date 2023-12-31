# Are Emergent Abilities in Large Language Models just In-Context Learning? [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

> **Abstract:** Large language models have exhibited *emergent* abilities, demonstrating exceptional performance across diverse tasks for which they were not explicitly trained, including those that require complex reasoning abilities. The emergence of such abilities carries profound implications for the future direction of research in NLP, especially as the deployment of such models becomes more prevalent. However, one key challenge is that the evaluation of these abilities is often confounded by competencies that arise in models through alternative prompting techniques, such as in-context learning and instruction following, which also emerge as the models are scaled up. In this study, we provide the first comprehensive examination of these emergent abilities while accounting for various potentially biasing factors that can influence the evaluation of models. We conduct rigorous tests on a set of 18 models, encompassing a parameter range from 60 million to 175 billion parameters, across a comprehensive set of 22 tasks. Through an extensive series of over 1,000 experiments, we provide compelling evidence that emergent abilities can primarily be ascribed to in-context learning. We find no evidence for the emergence of reasoning abilities, thus providing valuable insights into the underlying mechanisms driving the observed abilities and thus alleviating safety concerns regarding their use.

Contact persons: Sheng Lu, Irina Bigoulaeva, Harish Tayyar Madabushi

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

## evaluation scores
See [evaluation_scores](https://github.com/UKPLab/on-emergence/blob/main/evaluation_scores.csv) for the evaluation scores of our experiments.

## output files
All of the output files associated to our experiments can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3931). We name each file in a *date_time* fashion (e.g., ```results-20230524_123238.json```), which is also the ```run_id``` of an experiment that can be found in [evaluation_scores](https://github.com/UKPLab/on-emergence/blob/main/evaluation_scores.csv).

## citation
Please use the following citation:

```
@misc{lu2023emergent,
      title={Are Emergent Abilities in Large Language Models just In-Context Learning?}, 
      author={Sheng Lu and Irina Bigoulaeva and Rachneet Sachdeva and Harish Tayyar Madabushi and Iryna Gurevych},
      year={2023},
      eprint={2309.01809},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## license

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
