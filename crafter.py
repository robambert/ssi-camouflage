import torchvision.models as models
from torch import unsqueeze, sort
from torch.nn.functional import softmax
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import pylab as plt
import cvxpy as cvx
import dccp

def get_approx_coefficients(source_image, target_image):
  m, n, _ = np.array(source_image).shape
  m_prime, n_prime, _ = np.array(target_image).shape
  in_max = max(_[1] for _ in source_image.getextrema())
  S = np.identity(m) * in_max
  CL = Image.fromarray(S).resize((m_prime,m), resample= Image.BILINEAR)
  CL = np.array(CL)/in_max
  CL = normalize(CL, axis = 1, norm = 'l1')
  S_2 = np.identity(n) * in_max
  CR = Image.fromarray(S_2).resize((n,n_prime), resample= Image.BILINEAR)
  CR = np.array(CR)/in_max
  CR = normalize(CR, axis = 1, norm = 'l1')
  return CL, CR

duck = Image.open('images/duck.jpeg')
wolf = Image.open('images/wolf.bmp')
CL, CR = get_approx_coefficients(duck,wolf)

def strong_attack(source_path, target_path):
    def get_coefs(m, n, m_prime, n_prime, in_max):
        S = np.identity(m) * in_max
        CL = Image.fromarray(S).resize((m_prime,m), resample= Image.BILINEAR)
        CL = np.array(CL)/in_max
        CL = normalize(CL, axis = 1, norm = 'l1')
        S_2 = np.identity(n) * in_max
        CR = Image.fromarray(S_2).resize((n,n_prime), resample= Image.BILINEAR)
        CR = np.array(CR)/in_max
        CR = normalize(CR, axis = 1, norm = 'l1')
        return CL, CR

    def get_pertubation(x,y,coef_matrix, in_max):
        epsilon = 100000
        upper_bound = in_max * epsilon
        m = x.shape[0]
        delta = cvx.Variable(m)
        constraints = [
            0 <= x + delta, x + delta <= in_max,
            cvx.atoms.norm_inf(coef_matrix*(x + delta) - y) <= upper_bound
            ]
        objective = cvx.Minimize(cvx.norm(delta, 2))
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver = "CVXOPT")
        print(problem.status)
        print(delta.value)
        return delta.value
    
    def ScaleFunc(image,h,w):
        transform = transforms.Compose(
            [transforms.Resize((h,w)), 
            # transforms.CenterCrop((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
            )
        return transform(image)

    # importing images
    source = Image.open(source_path)
    target = Image.open(target_path)
    # unpacking source and target sizes
    m, n, _ = np.array(source).shape
    m_prime, n_prime, _ = np.array(target).shape
    in_max = max(_[1] for _ in source.getextrema())
    # fetchin coef matrices
    CL, CR = get_coefs(m,n,m_prime, n_prime, in_max)
    delta_v = np.zeros((m, n_prime))
    source_prime = ScaleFunc(source, m, n_prime)
    for col in range(n_prime - 1):
        delta_v[:,col] = get_pertubation(
            source_prime[:,col],
            target[:,col],
            CL,
            in_max
            )
    A_star = np.absolute((source_prime + delta_v).astype(int))
    delta_h = np.zeros((m, n))
    for row in range(m-1):
        delta_h[row,:] = get_pertubation(
            source[row,:],
            A_star[row,:],
            CR,
            in_max
        )
    A = np.absolute((source + delta_h).astype(int))
    return A


