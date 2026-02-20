"""Display utilities for printing model information."""


def print_model_params(model, title="Model Parameters"):
    """
    Print sklearn model parameters in a formatted way.

    Args:
        model: sklearn model with coef_ and intercept_ attributes
        title: Header title
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    print(f"\n  weight (coef_): {model.coef_.flatten().tolist()}")
    print(f"  bias (intercept_): {model.intercept_.tolist()}")

    total_params = model.coef_.size + model.intercept_.size
    print(f"\n  Total parameters: {total_params}")
    print(f"{'='*50}\n")
