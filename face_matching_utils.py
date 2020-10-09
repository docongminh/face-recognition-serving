import numpy as np
from scipy.stats import norm
import math

threshold = 73.18799151798612
mu_0 = 89.6058
sigma_0 = 4.5451
mu_1 = 43.5357
sigma_1 = 8.83

def compare(fn_0, fn_1):
    shape= fn_0.shape[-1]
    x0 = np.divide(fn_0, np.stack([np.linalg.norm(fn_0, axis=-1)]*shape, 1))
    x1 = np.divide(fn_1, np.stack([np.linalg.norm(fn_1, axis=-1)]*shape, 1))
    cosine = np.dot(x0, x1.T)
    theta = np.arccos(cosine)
    theta = theta * 180 / math.pi
    prob = get_prob(theta)
    return prob, theta

def get_prob(theta):
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    # print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
    return face_dist_value
