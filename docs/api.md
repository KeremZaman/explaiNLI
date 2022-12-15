# API

## Attribution Calculation through API

```python

attr_config = AttributionConfig(AttributionMethods.InputXGradient, remove_pad_tokens=True,
                                             remove_cls_token=True, remove_sep_tokens=True, join_subwords=True,
                                             normalize_scores=True,
                                             forward_scoring= ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN,
                                             label_names=['entailment', 'neutral', 'contradiction'])

attribution = NLIAttribution(model_name=nli_model_name, config=attr_config)
     

```

Supported attribution methods are `InputXGradient`, `Saliency`, `Guided` 

Supported aggregation methods are `L2` and `MEAN` and supported output methods are `TOP_PREDIICTION` and `LOSS`.

## Visualize

After obtaining attribution scores, visualize the given number 
of instances starting from a particular instance as following:

```
attribution.visualize(start=5, num=50)
```

This example visualizes attribution maps of 50 instances starting from 6th instance.
Starting instance and number of instances to visualize are optional parameters.

## Load/Save

For saving calculated attributions, use

```python
attribution.save('<path_to_save>')
```

For loading pre-recorded attributions, use

```python
NLIAttributon.from_file('<path_to_attributions>')
```

Attributions are recorded in this format using `pickle`: 

```python
attr_dict = {
            'model_name': self.model_name,
            'config': self.config,
            'records': self.records,
            'kwargs': self.kwargs,
        }
```
