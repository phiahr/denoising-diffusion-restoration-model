import torch
import numpy as np
import torchvision.transforms as transforms

def uniform_row_conv1d_matrix(input_shape, kernel_len, padding = 0):
    conv_matrix = torch.zeros(size = (input_shape[1], input_shape[1]-2*padding))
    multiplier = 1/(kernel_len)
    for col in range(0, conv_matrix.shape[1]):
        start = int(np.max([col-kernel_len//2,0]))
        end = int(np.min([col+kernel_len//2+1, conv_matrix.shape[0]]))
        conv_matrix[start:end, col] = multiplier

    return conv_matrix

class Degradation(object):

  def __init__(self, device = 'cpu', mean = 0., std = 0.):
    self.device = device
    self.mean = mean
    self.std = std

  def __call__(self, input_images):
    """
    Apply a given degradation on input images
    """
    pass

  def multiply_Vt(self, input_images):
    """
    Multiply the input images by V^T: V^T * x
    """
    pass

  def multiply_V(self, input_images):
    """
    Multiply
    """
    pass

  def get_y_bar(self, y):
    pass

  def get_x_bar(self, x):
    return self.multiply_Vt(x)

  def get_x_from_x_bar(self, x_bar):
    return self.multiply_V(x_bar)


class Noising(Degradation):
  def __init__(self, device = 'cpu', mean = 0., std=.1, input_shape = (256,256)):
    super().__init__(mean=mean, std=std, device = device)
    # self.std = std
    # self.mean = mean
    self.input_shape = input_shape
    self.singulars = torch.ones(input_shape[0]*input_shape[1])

  def __call__(self, tensor):
    tensor = tensor.to(self.device)
    return tensor + torch.randn(tensor.size()).to(device=self.device) * self.std + self.mean

  def multiply_Vt(self, input_images):
    return input_images

  def multiply_V(self, input_images):
    return input_images

  def get_y_bar(self, y):
    return y
  

class UniformBlur(Degradation):

  def __init__(self, device = 'cpu', kernel_size = (9,9), input_shape = (256,256), mean = 0., std = 0., ZERO = 5e-2):
    super().__init__(mean = mean, std = std, device = device)
    self.input_shape = input_shape
    self.kernel_size = kernel_size
    # Initialise Art and Ac matrices
    Art = uniform_row_conv1d_matrix(input_shape, kernel_len = kernel_size[0])
    self.Ac = Art.transpose(0,1).to(self.device)

    # Compute SVD of Ac
    Uc, Sc, Vch = torch.linalg.svd(self.Ac)

    # Compute singular values
    singulars = torch.matmul(Sc.unsqueeze(-1), Sc.unsqueeze(0)).flatten()

    # Sort singular values, store permutation
    self.singulars , self.permutation = torch.sort(singulars, descending = True)

    # Remove singulars < ZERO
    self.singulars[torch.where(self.singulars < ZERO)] = 0

    # Store elements of SVD of Ac
    self.Vc = Vch.transpose(0,1).to(self.device)
    self.Uc = Uc
    self.Sc = Sc


  def __call__(self, input_images):
    # Conmpute Ar_t
    Ar_t = self.Ac.transpose(0,1)

    # Compute blurred image as Ac * X * Ar_t
    blurred = torch.matmul(torch.matmul(self.Ac.unsqueeze(0).to(self.device), input_images.to(self.device)), Ar_t.unsqueeze(0).to(self.device))

    # Add noise
    noise = torch.randn(blurred.size()).to(self.device) * self.std + self.mean

    return blurred  + noise

  def __repr__(self):
    return self.__class__.__name__ + f"(kernel_size = {self.kernel_size})"

  def multiply_Vt(self, input_images):
    """
    Multiply the input images by V^T: V^T * x, input should have a batch format
    """
    # Get Vr transposed 
    Vrt = self.Vc.transpose(0,1)

    # Reshape images to match dimensions
    input_images = input_images.view(input_images.shape[0], 3, *self.input_shape)

    # Multiplication Vr^T * X * Vc which is effectively (Vc * X * Vr)^T) 
    product = torch.matmul(torch.matmul(Vrt.unsqueeze(0).to(self.device), input_images), self.Vc.unsqueeze(0).to(self.device)).flatten(start_dim = 2)

    # Apply permutation 
    permuted = product[:,:, self.permutation]
    return permuted

  def multiply_V(self, input_images):
    """
    Multiply the input images by V: V * x, input should have a batch format
    """
    Vrt = self.Vc.transpose(0,1)
    # Reshape images to match dimensions
    input_images = input_images.view(input_images.shape[0], 3, self.input_shape[0]**2)

    # Inverse permutation    
    permuted = torch.zeros_like(input_images)
    permuted[:,:,self.permutation] = input_images
    permuted = permuted.view(input_images.shape[0], 3, *self.input_shape)

    # Multiplication Vc * X * Vr^T which is effectively (Vr^T * X * Vc^T)^T)
    product = torch.matmul(torch.matmul(self.Vc.unsqueeze(0).to(self.device), permuted), Vrt.unsqueeze(0).to(self.device)).flatten(start_dim = 2)

    return product.view(*input_images.shape[:2], *self.input_shape)

  def multiply_Ut(self, input_images):
    """
    Multiply by U^T, input should have a batch format
    """
    # Reshape images to match dimensions
    input_images = input_images.view(input_images.shape[0], 3, *self.input_shape)

    # Get Ur transposed
    Urt = self.Uc.transpose(0,1)

    # Multiplication Ur^T * X * Uc which is effectively (Ur * x * Uc)^T in flattened form)
    product = torch.matmul(torch.matmul(Urt.unsqueeze(0), input_images), self.Uc.unsqueeze(0)).flatten(start_dim = 2)

    # Apply permutation
    permuted = product[:,:,self.permutation]
    return permuted

  def multiply_U(self, input_images):
    """
    Multiply by U, input should have a batch format
    """
    # Get Ur transposed
    Urt = self.Uc.transpose(0,1)

    # Reshape images to match dimensions
    input_images = input_images.flatten(start_dim = 2)

    # Inverse permutation
    permuted = torch.zeros_like(input_images)
    permuted[:,:,self.permutation] = input_images
    permuted = permuted.view(input_images.shape[0], 3, *self.input_shape)

    # Multiplication Uc * X * Ur^T which is effectively (Ur^T * X * Uc^T)^T in flattened form)
    product = torch.matmul(torch.matmul(self.Uc.unsqueeze(0).to(self.device), permuted), Urt.unsqueeze(0).to(self.device)).flatten(start_dim = 2)
    return product.view(*input_images.shape[:2], *self.input_shape)

  def get_y_bar(self, y):
    """
    Get y_bar from y by multiplying by Sigma^-1 * U^T, input should have a batch format
    """
    # multiply by Ut, will return permuted 1x3x256**2
    y = self.multiply_Ut(y)
    # Divide by singular values which is equivalent to multiplying by Sigma^-1
    y_bar = y / self.singulars.unsqueeze(0)
    return y_bar
