#ChatGPT generated code with some modifications
import torch

def expected_calibration_error(probs, labels, n_bins=15):
    """
    Compute the Expected Calibration Error (ECE) for a classification model.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape (N, C), where N is the number of samples,
                              and C is the number of classes.
        labels (torch.Tensor): True labels of shape (N,).
        n_bins (int): Number of bins for calibration.

    Returns:
        float: The ECE score.
    """
    # Ensure inputs are tensors
    # probs = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs
    # labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # Get predicted confidence and predicted class
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    # Initialize bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0).to(probs.device)

    # Calculate ECE
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()  # Fraction of samples in bin

        if prop_in_bin.item() > 0:  # Avoid division by zero
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

# Example Usage
if __name__ == "__main__":
    # Example predictions and labels
    probs = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    labels = torch.tensor([0, 1, 0, 1])

    ece = expected_calibration_error(probs, labels, n_bins=10)
    print(f"Expected Calibration Error: {ece:.4f}")
