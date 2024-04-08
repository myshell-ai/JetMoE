# Featurzing Function

This is repo is a fork of [JetMoE](https://github.com/myshell-ai/JetMoE). 

## Installation

Download this repo into your local machine
```
$ cd <your workspace>
$ git clone https://github.com/fanminshi/FeaturizingFunctions.git
$ cd FeaturizingFunctions
```

We first install this pkg so that its module can be accessible outside of this repo.

```
$ pip install -e .
```

Load modules from the installed pkg.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from jetmoe import JetMoEForCausalLM, JetMoEConfig, JetMoEForSequenceClassification

AutoConfig.register("jetmoe", JetMoEConfig)
AutoModelForCausalLM.register(JetMoEConfig, JetMoEForCausalLM)
AutoModelForSequenceClassification.register(JetMoEConfig, JetMoEForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained('jetmoe/jetmoe-8b')
model = AutoModelForCausalLM.from_pretrained('jetmoe/jetmoe-8b')
```





