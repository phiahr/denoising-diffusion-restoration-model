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
    self._singulars = torch.ones(input_shape[0]*input_shape[1])

  def __call__(self, tensor):
    return tensor.to(device=self.device) + torch.randn(tensor.size()).to(device=self.device) * self.std + self.mean

  def multiply_Vt(self, input_images):
    return input_images

  def multiply_V(self, input_images):
    return input_images

  def get_y_bar(self, y):
    return y
    
  def get_singulars(self, batch_size):
    return self._singulars.repeat(batch_size*3).view((batch_size, 3, *self.input_shape))
  
class SuperResolution(Degradation):
    def __init__(self, kernel_size, image_size, channels, mean=0.0, std=0.0, device=torch.device("cpu")):
        super().__init__(mean=mean, std=std, device = device)
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.channels = channels
        self.num_patches = image_size // kernel_size
        self.device = device

        k = (torch.ones(kernel_size ** 2) / (kernel_size ** 2)).reshape(-1,1)
        k_T = k.transpose(0, 1).to(device)

        self.U, self.s, self.V = torch.svd(k_T, some=False)

        self._singulars = self.s.repeat(self.num_patches**2).repeat(channels)

    def __call__(self, tensor):
      output_image = torch.zeros(tensor.shape[0], self.channels, self.image_size//self.kernel_size, self.image_size//self.kernel_size)
      vt = self.multiply_by_Vt(tensor.to(device=self.device)).view(tensor.shape[0], -1)
      temp = self._singulars * vt[:,:3*self.num_patches**2]
      temp = self.U * temp.view(tensor.shape[0], self.channels, self.image_size//self.kernel_size, self.image_size//self.kernel_size)
      output_image = temp.reshape(tensor.shape[0],self.channels, self.image_size//self.kernel_size, self.image_size//self.kernel_size)

      noise = torch.randn(output_image.size()).to(self.device) * self.std + self.mean
      output_image += noise
      return output_image

    def Ut(self, tensor_image):
        UT = torch.full_like(tensor_image, self.U.item()).reshape(self.channels, -1)
        return UT * tensor_image.reshape(self.channels, -1)

    def multiply_by_Vt(self, tensor_image):

        input = tensor_image.reshape(tensor_image.shape[0],self.channels, self.num_patches, self.kernel_size, self.num_patches, self.kernel_size).to(self.device)
        input = input.permute(0,1, 2, 4, 3, 5)

        P1 = input.reshape(tensor_image.shape[0],self.channels, -1, self.kernel_size**2)
        # transpose P1 along its second two dimensions
        P1_T = P1.permute(0, 1, 3, 2)

        # print("my Patches", P1)

        res = torch.matmul(self.V.t(), P1_T)
        # print("multiply by vt", res)

        # one_channel_res = torch.matmul(self.V.t(), P1[0].t())

        res = res.permute(0,1, 3,2)
        # print(res)
        # res = res.reshape(channels,-1,kernel_size**2)
        res = res.reshape(tensor_image.shape[0],self.channels,-1,self.kernel_size**2).permute(0,1, 3,2)
        # print(res)

        flattened_tensor = torch.zeros((tensor_image.shape[0],self.channels*self.image_size**2)).to(self.device)

        # Get the first part of the flattened tensor
        # print(res[:,:,0,:])
        first_values = res[:, :, 0, :].reshape(tensor_image.shape[0],-1)
        flattened_tensor[:,:first_values.shape[-1]] = first_values

        # Calculate the indices for the second part of the flattened tensor
        indices = torch.arange(res.shape[-1], self.image_size**2, self.kernel_size**2-1)

        # Reshape 'res' to make it suitable for indexing
        res_reshaped = res[:,:, 1:, :].permute(0,1, 3, 2).contiguous().view(tensor_image.shape[0],self.channels, -1)
        # print("flat", flattened_tensor)
        flattened_tensor[:,]
        # print(res_reshaped)

        # Assign values to the second part of the flattened tensor using indexing
        # flattened_tensor[:,indices] = res_reshaped[:, :, :indices.size(0)]
        
        flattened_tensor[:, first_values.shape[-1]:] = res_reshaped.reshape(tensor_image.shape[0],-1)

        return flattened_tensor.reshape(tensor_image.shape[0],self.channels, tensor_image.shape[-1], tensor_image.shape[-1])
    
    def multiply_by_V(self, tensor_image):
        flattened_tensor = tensor_image.flatten().reshape(tensor_image.shape[0],-1, self.image_size**2//self.kernel_size**2).to(self.device)
        P1 = torch.zeros((tensor_image.shape[0],self.channels, self.kernel_size ** 2, self.image_size ** 2 // self.kernel_size ** 2)).to(self.device)
        P1[:,:, 0, :] = flattened_tensor[:,:self.channels]
        P1 = P1.permute(0, 1, 3, 2)

        remaining_values = flattened_tensor.flatten().reshape(tensor_image.shape[0],-1)[:,self.image_size ** 2 // self.kernel_size**2 *self.channels:]
        remaining_values = remaining_values.reshape(tensor_image.shape[0],self.channels, -1, self.kernel_size**2-1)
        P1[:, :, :, 1:] = remaining_values
        # print("patches", P1)

        final = torch.matmul(P1, self.V)
        # print(final)
        # print("multiply by v", final)

        final = final.reshape(tensor_image.shape[0],self.channels, self.num_patches, self.num_patches, self.kernel_size, self.kernel_size)
        final = final.permute(0,1, 2,4,3,5)

        final = final.reshape(tensor_image.shape[0],self.channels,self.image_size**2).flatten()
        return final.reshape(tensor_image.shape[0],self.channels, tensor_image.shape[-1], tensor_image.shape[-1])

    def get_y_bar(self, y):
        tmp = self.Ut(y)

        y_bar = tmp / self._singulars.repeat(y.shape[0])[:tmp.shape[-1]]

        y_bar_full_size = torch.zeros((y.shape[0], self.channels*self.image_size**2)).to(self.device)
        # y_bar_full_size = y_bar.reshape((y.shape[0], -1))
        y_bar_full_size[:, :self.channels*self.num_patches**2] = y_bar.view((y.shape[0],-1))
        y_bar_full_size = y_bar_full_size.reshape((y.shape[0], self.channels, self.image_size, self.image_size))

        return y_bar_full_size
    
    def get_x_bar(self, x):
        return self.multiply_by_Vt(x)
    
    def get_x_from_x_bar(self, x_bar):
        return self.multiply_by_V(x_bar)

    def get_singulars(self, batch_size):
      singulars = torch.zeros((batch_size, self.channels*self.image_size**2)).to(self.device)
      singulars[:,:self.channels*self.num_patches**2] = self._singulars.repeat(batch_size).view((batch_size,-1))#.view(y.shape)
      singulars = singulars.reshape((batch_size, self.channels, self.image_size, self.image_size))
      return singulars
    

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
    _singulars = torch.matmul(Sc.unsqueeze(-1), Sc.unsqueeze(0)).flatten()

    # Sort singular values, store permutation
    self._singulars , self.permutation = torch.sort(_singulars, descending = True)

    # Remove _singulars < ZERO
    self._singulars[torch.where(self._singulars < ZERO)] = 0

    # Store elements of SVD of Ac
    self.Vc = Vch.transpose(0,1).to(self.device)
    self.Uc = Uc
    self.Sc = Sc


  def __call__(self, input_images):
    # Conmpute Ar_t
    Ar_t = self.Ac.transpose(0,1)

    # Compute blurred image as Ac * X * Ar_t
    blurred = torch.matmul(torch.matmul(self.Ac.unsqueeze(0).to(self.device), input_images.to(device=self.device)), Ar_t.unsqueeze(0).to(self.device))

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
    y_bar = y / self._singulars.unsqueeze(0)
    return y_bar.view((y.shape[0], y.shape[1], *self.input_shape))

  def get_singulars(self, batch_size):
    return self._singulars.repeat(batch_size*3).view((batch_size, 3, *self.input_shape))