import numpy as np

def mds(d, dim=2):
    """
    Multi-Dimensional Scaling (MDS) algorithm to compute 2D or 3D vectors
    that minimize the given distance matrix.
    
    Args:
        d (numpy.ndarray): NxN distance matrix
        dim (int): Dimension of space to project to, should be either 2 or 3 (default=2)
        
    Returns:
        x (numpy.ndarray): Nx2 or Nx3 vectors
        
    Raises:
        ValueError: If input distance matrix is not square or has size less than 3
        ValueError: If dimension of space to project to is not 2 or 3
    """
    if d.shape[0] != d.shape[1]:
        raise ValueError('Input distance matrix d should be square!')
    if d.shape[0] < 3:
        raise ValueError('Matrix size should be at least 3')
    if dim not in [2, 3]:
        raise ValueError('Dimension of space to project to should be either 2 or 3')
        
    N = d.shape[0]
    
    # Step 1: randomly assign x in 0-1, 0-1 or 0-1, 0-1, 0-1
    x = np.random.rand(N, dim)
    
    # Step 2: compute error
    def compute_mds_error(d, x, i):
        return np.sum((np.linalg.norm(x[i] - x, axis=1) - d[i])**2)
    
    E = np.zeros(N)
    for i in range(N):
        E[i] = compute_mds_error(d, x, i)
    
    # Step 3: Iterate
    v = np.cos(np.pi/4)
    points_org = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]])
    Np = points_org.shape[0]
    Enew = np.sum(E)
    iter = 0
    while ((iter == 0) or (Eorg > Enew)):
        iter += 1
        Eorg = Enew
        step = max(0.01, 0.1/(np.sqrt(iter)))
        points = step * points_org
        xorg = np.copy(x)
        for i in range(N):
            E[i] = compute_mds_error(d, x, i)
            Eloc = np.zeros(Np)
            for j in range(Np):
                x[i] = xorg[i] + points[j]
                Eloc[j] = compute_mds_error(d, x, i)
            ord = np.argmin(Eloc)
            x[i] = xorg[i] + points[ord]
            Enew = Enew - 2 * (E[i] - Eloc[ord])
            E[i] = Eloc[ord]
        print(iter, '(', step, '): ', Enew)
        
    return x