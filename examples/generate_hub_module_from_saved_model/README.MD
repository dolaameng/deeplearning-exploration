# Code Snippet to generate tensorflow hub from a saved_model

## Motivation:
- saved_model form is getting more and more popular when training/serving tensorflow models
- tensorflow hub module is convienient for model/code reuse

## Approach
- I haven't fully figured out how to convert a saved_model with variables to a hub module, with the main difficulty being that the variables from the saved_model need to be explicitly assigned when exporting them to hub module.
- On the other hand, it is easier to just freeze the saved_model first by converting all variables to constants (as part of graph now), and then converting it to a hub.
- pros: easier to use, cons: impossible to fine tune the model
- The example codes here demostrate this approach.