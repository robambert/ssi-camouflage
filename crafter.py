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
        CL = Image.fromarray(S).resize((m, m_prime), resample= Image.BILINEAR)
        CL = np.array(CL)/in_max
        CL = normalize(CL, axis = 1, norm = 'l1')
        S_2 = np.identity(n) * in_max
        CR = Image.fromarray(S_2).resize((n_prime, n), resample= Image.BILINEAR)
        CR = np.array(CR)/in_max
        CR = normalize(CR, axis = 1, norm = 'l1')
        return CL, CR

    def get_pertubation(x,y,coef_matrix, in_max):
        epsilon = 100000
        upper_bound = in_max * epsilon
        m = x.shape[0]
        delta = cvx.Variable(m)
        # print(coef_matrix.shape)
        # print(x.shape)
        # print(y.shape)
        # print(delta.shape)

        constraints = [
            0 <= x + delta, x + delta <= in_max,
            cvx.atoms.norm_inf(coef_matrix*(x + delta) - y) <= upper_bound
            ]
        objective = cvx.Minimize(cvx.norm(delta, 2))
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver = "CVXOPT")
        # print(problem.status)
        # print(delta.value)
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

    # importing images as np array
    source_img = Image.open(source_path)
    target_img = Image.open(target_path)
    source = np.array(source_img)
    target = np.array(target_img)
    # unpacking source and target sizes
    m, n, _ = source.shape
    m_prime, n_prime, _ = target.shape
    print('Source %i x %i' %(m,n))
    print('Target %i x %i' %(m_prime, n_prime))
    in_max = max(_[1] for _ in source_img.getextrema())
    # fetchin coef matrices
    CL, CR = get_coefs(m,n,m_prime, n_prime, in_max)
    print('CL %i x %i'%CL.shape)
    print('CR %i x %i'%CR.shape)
    delta_v = np.zeros((m, n_prime,3))
    # print(delta_v.shape)
    source_prime_img = ScaleFunc(source_img, m, n_prime)
    source_prime = np.array(source_prime_img)
    # print(source_prime.shape)
    # print(target.shape)
    for col in range(n_prime - 1):
        for color in range(3):
            print('Currently at column %i for channel %i'%(col, color))
            delta_v[:,col, color] = get_pertubation(
                source_prime[color, :, col],
                target[:,col, color],
                CL,
                in_max
                )
    A_star = np.absolute((source_prime + delta_v).astype(int))
    delta_h = np.zeros((m, n))
    for row in range(m-1):
        for color in range(3):
            print('Currently at row %i for channel %i'%(col, color))
            delta_h[row,:,color] = get_pertubation(
                source[row,:, color],
                A_star[row,:, color],
                CR,
                in_max
            )
    A = np.absolute((source + delta_h).astype(int))
    return A

source_path = 'images/duck.jpeg'
target_path = 'images/eiffel.jpg'

print(strong_attack(source_path, target_path))
# source_img = Image.open(source_path)
# target_img = Image.open(target_path)
# source = np.array(source_img)
# target = np.array(target_img)

# print(source.shape)
# print(target.shape)
