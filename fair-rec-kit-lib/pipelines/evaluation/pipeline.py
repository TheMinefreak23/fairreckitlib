from abc import abstractmethod


class EvaluationPipeline():
    def __init__(self, api_name, factory):
        self._api_name = api_name
        self._factory = factory

    def run(self, metrics, callback, **kwargs):
        callback.on_begin_pipeline(self._api_name)

        self._run_batch(metrics, callback, **kwargs)

        callback.on_end_pipeline(self._api_name)

    def _run_batch(self, metrics, callback, **kwargs):
        for metric_name in metrics:
            self._run_metric(metric_name, metrics[metric_name], callback, **kwargs)

    @abstractmethod
    def _run_metric_test(self, metric, callback, **kwargs):
        raise NotImplementedError()

    def _run_metric(self, metric_name, params, callback, **kwargs):
        metric = self._create_metric(metric_name, params, callback)
        self._run_metric_test(metric, callback, **kwargs)

    def _create_metric(self, metric_name, params, callback):
        callback.on_create_model(metric_name, params)
        return self._factory[metric_name]['create'](params)