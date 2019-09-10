from typing import Callable


class CachablePipelineStep:
    def load(self):
        pass

    def __call__(self):
        pass

    def save(self, results):
        pass


def build_callable_that_loads_from_cache_or_runs_step(use_cache: bool, pipeline_step):
    def new_callable(*args, **kwargs):
        if use_cache:
            try:
                return pipeline_step.load()
            except FileNotFoundError:
                results = pipeline_step(*args, **kwargs)
                pipeline_step.save(results)
                return results

        else:
            return pipeline_step()

    return new_callable


def build_callable_that_loads_from_cache_or_runs_func(
        use_cache: bool,
        load_function: Callable,
        save_function: Callable,
        main_function: Callable,
        *args, **kwargs
):
    """
    :param use_cache: specify whether to return the results from load_function or to execute and
        return results from main_function.
    :param load_function: Function that loads the cached result.
        Should throw FileNotFoundError if cached data not found,
        at which point the main_function executes and its results are saved.
        Is not run if use_cache=False.
    :param save_function: A function that should save the results from main_function.
        Is not run if use_cache=False.
    :param main_function: main function to perform. Runs if FileNotFoundError thrown from load_function.
        Always ran if use_cache=False.
    :param args: args to pass to main_function
    :param kwargs: kwargs to pass to main_function
    :return: a function that returns the cached results, executes the main_function if no results, or executes the
        main_function if use_cache=False passed
    """
    def new_callable():
        if use_cache:
            try:
                return load_function()
            except FileNotFoundError:
                results = main_function(*args, **kwargs)
                save_function(results)
                return results
        else:
            return main_function(*args, **kwargs)
    return new_callable


pipeline_example = build_callable_that_loads_from_cache_or_runs_func(
    use_cache=True,
    load_function=lambda: [1, 2, 3],
    save_function=lambda results: print('saved :', results),
    main_function=lambda: "no arg main",
)

