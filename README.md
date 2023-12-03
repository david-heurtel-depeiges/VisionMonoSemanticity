# VisionMonoSemanticity
A repository trying to implement Anthropic monosemantic approach in language transformers but for vision models

## Report 

An early report about this project can be found [here](https://david-heurtel-depeiges.github.io/monosemantic)

## Using the code

This code is still in development and is not yet ready to be used. However, if you want to try it, you can clone the repository. You should modify (at least):
- train.py to point to where you store ImageNet
- model.py if you want to use other models

## TODOs
- [ ] Add a requirements.txt
- [ ] Transform that in a package for ease of use
- [ ] Ensure that the code would function for other models
- [ ] Add a demo notebook
- [ ] Manage callbacks for checkpointing in a specific directory (different from the logging directory)
- [ ] Clarify where configs files should be stored and how they should be retrieved
