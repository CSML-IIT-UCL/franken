import torch


@torch.no_grad()
def conjugate_gradient(
    A, b: torch.Tensor, x0: torch.Tensor | None = None, max_iter=50, tol=1e-6
):
    if x0 is not None:
        x = x0
        r = b - (A @ x)
    else:
        r = b
        x = torch.zeros_like(b)

    p = r
    rsold = torch.sum(r * r)

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rsold / (torch.dot(p.view(-1), Ap.view(-1)) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.view(-1), r.view(-1))
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x
