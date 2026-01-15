import torch
import torch.nn.functional as F

def pgd_counterfactual(
        model,
        x,
        true_label=None,
        target_label=None,
        epsilon=0.3,
        step_size=0.01,
        num_steps=40,
        clamp_min=0.0,
        clamp_max=1.0,
        targeted=False
):
    """
    Compute a PGD-based counterfactual for a single MNIST input x.

    Args:
        model         : instance of Mnist6xS (already in eval() mode, on same device as x)
        x             : input tensor of shape (1, 784) or (1, 1, 28, 28), values in [0,1]
        true_label    : scalar long tensor with original class (for sanity checks, optional)
        target_label  : scalar long tensor with desired class (required if targeted=True)
        epsilon       : L_inf radius of allowed perturbation
        step_size     : PGD step size (alpha)
        num_steps     : number of PGD iterations
        clamp_min/max : pixel range after perturbation
        targeted      : if True, push towards target_label; else push away from true_label

    Returns:
        x_cf          : counterfactual input tensor
        y_pred        : predicted label for x_cf
    """

    device = x.device
    model.eval()

    # Ensure shape is (1, 784) for this fully-connected network
    if x.dim() == 4:            # (N, C, H, W) -> (N, 784)
        x_flat = x.view(x.size(0), -1)
    else:
        x_flat = x.clone()

    x_orig = x_flat.detach()
    x_adv = x_orig.clone()

    if targeted and target_label is None:
        raise ValueError("target_label must be provided for targeted PGD counterfactual.")

    if targeted:
        y = target_label.view(1).to(device)
    else:
        if true_label is None:
            # If no true label, take current model prediction as the 'true'
            with torch.no_grad():
                y = model(x_orig).argmax(dim=1)
        else:
            y = true_label.view(1).to(device)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)  # shape (1, 10)

        if targeted:
            # For a targeted counterfactual, MINIMIZE loss towards target class
            loss = F.cross_entropy(logits, y)
            # PGD update: gradient descent on loss (move towards target)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - step_size * torch.sign(grad)
        else:
            # For untargeted, MAXIMIZE loss of current class (move away)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + step_size * torch.sign(grad)

        # Project into L_inf ball around x_orig
        perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = x_orig + perturbation

        # Keep in valid pixel range
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max).detach()

    # Final prediction
    with torch.no_grad():
        y_pred = model(x_adv).argmax(dim=1)

    return x_adv, y_pred

def pgd_counterfactual_multi_restart(
        model, x, true_label, target_label,
        epsilon=0.3, step_size=0.03, num_steps=100,
        n_restarts=5, clamp_min=0.0, clamp_max=1.0
):
    device = x.device
    model.eval()
    x_orig = x.detach()

    best_x_adv = x_orig.clone()
    best_success = False

    y_target = target_label.view(1).to(device)

    for _ in range(n_restarts):
        # random start in L_inf ball
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

        for _ in range(num_steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y_target)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - step_size * torch.sign(grad)

            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + delta, clamp_min, clamp_max).detach()

        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        if pred.item() == y_target.item():
            best_x_adv = x_adv
            best_success = True
            break  # early exit if you just need one CF

    return best_x_adv, best_success

def pgd_counterfactual_binary(
        model,
        x,
        true_label=None,
        target_label=None,
        epsilon=0.3,
        step_size=0.01,
        num_steps=40,
        clamp_min=0.0,
        clamp_max=1.0,
        targeted=False,
):
    device = x.device
    model.eval()

    x_orig = x.detach()
    x_adv = x_orig.clone()

    # Determine labels
    if targeted:
        if target_label is None:
            raise ValueError("target_label must be provided for targeted PGD.")
        y = torch.as_tensor(target_label, device=device).view(-1).float()
    else:
        if true_label is None:
            with torch.no_grad():
                logits0 = model(x_orig)
                y = (logits0 > 0).long().view(-1).float()
        else:
            y = torch.as_tensor(true_label, device=device).view(-1).float()

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    for _ in range(num_steps):
        with torch.enable_grad():
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = bce(logits.view(-1), y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        # PGD update
        if targeted:
            x_adv = x_adv - step_size * torch.sign(grad)
        else:
            x_adv = x_adv + step_size * torch.sign(grad)

        # Project and clamp
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, clamp_min, clamp_max).detach()

    with torch.no_grad():
        logits_cf = model(x_adv)
        y_pred = (logits_cf > 0).long().view(-1)

    return x_adv, y_pred


