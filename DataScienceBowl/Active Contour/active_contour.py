######################################################################
# This routine is an implementation of the active contour model
# following Chan, T. and Vese, L.(2001): Active Contours without Edges
######################################################################
import numpy as np
import skfmm
from PIL import Image


def load_image_asarray(filename, name):
    img = Image.open(filename).convert('LA')
    img.load()
    img_name = np.asarray(img, dtype="int32")
    return img_name


heart = load_image_asarray(r"DataScienceBowl/Active Contour/ROI_image.png", "heart")


def evolve_contour(lv, roi, deltaT=0.1, alpha1=1, alpha2=0.5, alpha3=0.25, eps=1/np.pi, eta=1e-5):
    """
    evolve_contour performs an active contour algorithm on a level set curve,
    specifically on the zero level of a signed distance function. The zero-level
    is represented by discrete points on a grid.
    Each point is evolved in the direction where some energy function decreases most
    :param lv: 64 x 64 binary array
    :param roi: 64 x 64 region of interest (gray scale)
    :param deltaT: the step size in time
    :param alpha1, alpha2, alpha3: parameters for energy components
    :param eps: parameter for the function approximating the delta_0 function
    :return: evolved level set function
    """
    # error handling: inputs must be 64 x 64
    if (np.shape(lv) != (64, 64) | np.shape(roi)!= (64, 64)):
        raise TypeError("lv and roi must be 64 x 64")

    # Initialize phi as a signed distance function which looks like a party hat. It
    # computes for each point in the 64 x 64 region of interest its distance to
    # the closest contour point in the binary image LV.
    phi = lv
    phi[phi == 1] = -1
    phi[phi == 0] = 1
    phi = -skfmm.distance(phi)
    # we will store the initialization of phi again in an extra variable because
    # we have to recall it every update
    phi0 = phi
    ## START ENERGY OPTIMIZATION
    convergence = False
    while convergence == False:

        # 1. compute all finite differences
        # this will be done by the divergence function, so forget about this step

        # 2. compute averages inside and outside contour
        # determine which pixels are inside the contour and which are outside
        # a pixel at (i,j) is inside the contour if phi(i,j) > 0
        roi_inside = roi(phi > 0)
        roi_outside = roi(phi < 0)
        c1 = np.mean(roi_inside)
        c2 = np.mean(roi_outside)

        # 3. Compute divergence
        div = get_div(phi)
        old_phi = phi

        # 4. Evolve contour
        phi = phi + deltaT * (delta_eps(phi) * (div + alpha2 * (roi - c1)^2 \
                                                - alpha2 * (roi - c2)^2 \
                                                - 2* alpha3 * (phi - phi0)))

        # 5. stop if phi has converged
        if np.linalg.norm(phi - old_phi, 'fro') < eta:
            convergence = True
    return phi


def get_contour(s):
    """
    get_contour takes binary array LV array and extracts the pixels on the contour
    The result is again a binary image where the inner part of the LV is also black.
    Only the contour appears white
    :param s: binary array
    :return: binary array
    """
    # loop through every pixel of the binary image and "keep" only those
    # which are foreground and neighbored by a background pixel
    bound = np.zeros(np.shape(s))
    for k, l in np.ndindex(np.shape(s)):
        if 0 < k < (np.shape(s)[0]-1) and 0 < l < (np.shape(s)[1]-1):
            if s[k, l] == 1 and \
                    (s[k-1, l] == 0 or s[k+1, l] == 0 or s[k, l-1] == 0 or s[k, l+1] == 0):
                bound[k, l] = 1
    return bound


def hessian(x):
    """
    :param x: numpy array
       - x : ndarray
    :return: hessian of x
           hessian[0,0] = deriv_yy
           hessian[0,1] = deriv_yx
           hessian[1,0] = deriv_xy
           hessian[1,1] = deriv_xx
    """
    # compute the hessian and lay it out in a format which is sensible
    x_grad = np.gradient(x)
    h = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            h[k, l, :, :] = grad_kl
    return h


def get_div(x):
    """
    :param x: numpy array
    :return: divergence of x
    """
    # compute matrices of derivatives in x- and y- direction
    g = np.gradient(x)
    dy = g[0]
    dx = g[1]
    # compute matrix of norm of gradient
    gnorm = np.sqrt(np.power(g[0], 2) + np.power(g[1], 2))
    # compute hessian
    h = hessian(x)
    dyy = h[0, 0]
    dyx = h[0, 1]
    dxx = h[1, 1]
    # compute divergence
    div = (dxx + dyy)/gnorm - (dx*(dxx+dyx) + dy*(dyx + dyy))/np.power(gnorm, 3)
    return div

def delta_eps(x, eps):
    return 1/(eps * np.pi * (1 + np.power((x/eps),2)))