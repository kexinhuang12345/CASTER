from __future__ import print_function
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
torch.manual_seed(1)    # reproducible
np.random.seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

    
class dde_NN_Large_Predictor(nn.Sequential):
	'''
	first draft
	input dimension:
		X_pair: batch_size x eta x 1
		X_entries: eta x eta , f = # substructures
	'''
	def __init__(self, **config):
		super(dde_NN_Large_Predictor, self).__init__()
		self.input_dim = config['input_dim']
		self.num_class = config['num_class']
		self.lambda3 = config['lambda3']        
		self.encode_fc1_dim = config['encode_fc1_dim']
		self.encode_fc2_dim = config['encode_fc2_dim']
		self.decode_fc1_dim = config['decode_fc1_dim']
		self.decode_fc2_dim = config['decode_fc2_dim']
		self.predict_dim = config['predict_dim']
		self.predict_out_dim = config['predict_out_dim']
		self.mag_factor = config['magnify_factor']        
		# encoder: two layer NN
		self.encoder = nn.Sequential(
			nn.Linear(self.input_dim, self.encode_fc1_dim),
			nn.ReLU(True),
			nn.Linear(self.encode_fc1_dim, self.encode_fc2_dim)
		)
		# decoder: two layer NN
		self.decoder = nn.Sequential(
			nn.Linear(self.encode_fc2_dim, self.decode_fc1_dim),
			nn.ReLU(True),
			nn.Linear(self.decode_fc1_dim, self.decode_fc2_dim)
		)
		# predictor: eight layer NN
		self.predictor = nn.Sequential(
            # layer 1
			nn.Linear(self.input_dim, self.predict_dim),
			nn.ReLU(True),
            # layer 2
			nn.BatchNorm1d(self.predict_dim),
			nn.Linear(self.predict_dim, self.predict_dim),
			nn.ReLU(True),
            # layer 3
			nn.BatchNorm1d(self.predict_dim),
			nn.Linear(self.predict_dim, self.predict_dim),
			nn.ReLU(True),
            # layer 4
			nn.BatchNorm1d(self.predict_dim),
			nn.Linear(self.predict_dim, self.predict_dim),
			nn.ReLU(True),
            # layer 5
			nn.BatchNorm1d(self.predict_dim),
			nn.Linear(self.predict_dim, self.predict_dim),
			nn.ReLU(True),
            # layer 6
			nn.BatchNorm1d(self.predict_dim),
			nn.Linear(self.predict_dim, 64),
			nn.ReLU(True),
            # output layer
			nn.Linear(64, self.predict_out_dim)
		)

	def dictionary_encoder(self, Z_D, Z_f, v_D):
		'''
		:param v_D: batch_size x eta
		:param Z_D: batch_size x encode_fc2_dim
		:param Z_f: encode_fc2_dim x eta
		:return: sparse code X_o: batch_size x eta
		'''       
		DTD = torch.matmul(Z_f, Z_f.transpose(2, 1))  # D is Dictionary;  D^T D encode_dim x eta
		DTD_inv = torch.inverse(DTD + self.lambda3 * torch.eye(self.input_dim).cuda())  # (D^T D + \lambda2 I )^{-1} D^T D, eta x eta
		DTD_inv_DT = torch.matmul(DTD_inv, Z_f)  
		# (D^T D + lambda I)^{-1} D^T,  eta x encode_dim
		# assert DTD_inv_DT.requires_grad == True # check
		r = Z_D[:,None,:].matmul(DTD_inv_DT.transpose(2, 1)).squeeze(1) # batch_size x eta    
		return r

	def forward(self, v_D):
		'''
		:param v_D: batch_size x eta, multi-hot vector
		:return: recon, score, code
		'''
		_, eta = v_D.shape
		# encode
		#print(v_D.dtype)        
		Z_D = self.encoder(v_D.cuda())
		# print(Z_D)
		Z_f = self.encoder(torch.eye(eta).cuda())
		Z_f = Z_f.mul(v_D[:,:,None].cuda()) 
		# print(Z_f)                
		# decode
		v_D_hat = self.decoder(Z_D)
		recon = torch.sigmoid(v_D_hat)
		# dictionary learning
		code = self.dictionary_encoder(Z_D, Z_f, v_D)
		score = self.predictor(self.mag_factor * code)
		return recon, code, score, Z_f, Z_D
    
   