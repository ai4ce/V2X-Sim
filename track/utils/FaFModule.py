import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.FaFModel import FaFNet,FaFMIMONet,FeatEncoder,FaFMGDA,FaFMIMONetKD
from utils.detection_util import *
from utils.min_norm_solvers import MinNormSolver
import numpy
import matplotlib.pyplot as plt
from data.obj_util import coor_to_vis

class FaFModule(object):
	def __init__(self, model,config,optimizer, criterion):
		self.MGDA = config.MGDA
		if self.MGDA:
			self.encoder = model[0]
			self.head = model[1]
			self.optimizer_encoder = optimizer[0]
			self.optimizer_head = optimizer[1]
			self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_encoder, milestones=[50, 100, 150, 200], gamma=0.5)
			self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_head, milestones=[50, 100, 150, 200], gamma=0.5)
			self.MGDA = config.MGDA
		else:
			self.model = model
			self.optimizer = optimizer
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
		self.criterion = criterion#{'cls_loss','loc_loss'}
		
		self.out_seq_len = config.pred_len
		self.category_num = config.category_num
		self.code_size = config.box_code_size
		self.loss_scale = None
		
		self.code_type = config.code_type
		self.loss_type=  config.loss_type
		self.pred_len = config.pred_len
		self.only_det = config.only_det
		if self.code_type in ['corner_1','corner_2','corner_3']:
			self.alpha = 1.
		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				self.alpha= 1.
				if not self.only_det:
					self.alpha = 1.
			else:
				self.alpha = 0.1
		self.config = config

	def resume(self,path):
		def map_func(storage, location):
			return storage.cuda()

		if os.path.isfile(path):
			if rank == 0:
				print("=> loading checkpoint '{}'".format(path))

			checkpoint = torch.load(path, map_location=map_func)
			self.model.load_state_dict(checkpoint['state_dict'], strict=False)


			ckpt_keys = set(checkpoint['state_dict'].keys())
			own_keys = set(model.state_dict().keys())
			missing_keys = own_keys - ckpt_keys
			for k in missing_keys:
				print('caution: missing keys from checkpoint {}: {}'.format(path, k))
		else:
				print("=> no checkpoint found at '{}'".format(path))


	def corner_loss(self,anchors,reg_loss_mask,reg_targets,pred_result):
		N = pred_result.shape[0]
		anchors = anchors.unsqueeze(-2).expand(anchors.shape[0],anchors.shape[1],anchors.shape[2],anchors.shape[3],reg_loss_mask.shape[-1],anchors.shape[-1])
		assigned_anchor = anchors[reg_loss_mask]
		assigned_target = reg_targets[reg_loss_mask]
		assigned_pred = pred_result[reg_loss_mask]
		#print(assigned_anchor.shape,assigned_pred.shape,assigned_target.shape)
		#exit()
		pred_decode = bev_box_decode_torch(assigned_pred,assigned_anchor)
		target_decode = bev_box_decode_torch(assigned_target,assigned_anchor)
		pred_corners = center_to_corner_box2d_torch(pred_decode[...,:2],pred_decode[...,2:4],pred_decode[...,4:])
		target_corners = center_to_corner_box2d_torch(target_decode[...,:2],target_decode[...,2:4],target_decode[...,4:])
		loss_loc = torch.sum(torch.norm(pred_corners-target_corners,dim=-1)) / N

		return loss_loc

	def loss_calculator(self,result,anchors,reg_loss_mask,reg_targets,labels,N,motion_labels = None,motion_mask=None):
		loss_num =0
		# calculate loss
		weights = torch.Tensor([0.005, 1.0, 1.0, 1.0, 1.0]).cuda().double()
		loss_cls = torch.sum(self.criterion['cls'](result['cls'],labels)) /N
		loss_num += 1
		#loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N

		#Motion state
		if not motion_labels is None:
			loss_motion = torch.sum(self.criterion['cls'](result['state'],motion_labels)) /N
			loss_num += 1

		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1,reg_loss_mask.shape[-1])).size(0)
		#print(loss_mask_num)
		#print(torch.sum(reg_targets[:,:,:,:,0][reg_loss_mask[:,:,:,:,2]]))

		if self.code_type in ['corner_1','corner_2','corner_3']:
				target = reg_targets[reg_loss_mask].reshape(-1,5,2)
				flip_target = torch.stack([target[:,0],target[:,3],target[:,4],target[:,1],target[:,2]],dim=-2)
				pred = result['loc'][reg_loss_mask].reshape(-1,5,2)
				t = torch.sum(torch.norm(pred-target,dim=-1),dim=-1)
				f = torch.sum(torch.norm(pred-flip_target,dim=-1),dim=-1)
				loss_loc = torch.sum(torch.min(t,f)) / N
				

		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors,reg_loss_mask,reg_targets,result['loc'])
					loss_num += 1
				elif self.config.pred_type in ['motion','center']:

					###only center/motion for pred
					
					loss_loc_1 = self.corner_loss(anchors,reg_loss_mask[...,0][...,[0]],reg_targets[...,[0],:],result['loc'][...,[0],:])
					pred_reg_loss_mask = reg_loss_mask[...,1:,:]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask #mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][...,1:,:][pred_reg_loss_mask],reg_targets[...,1:,:][pred_reg_loss_mask]) 
					loss_loc = loss_loc_1 + loss_loc_2
					loss_num += 2
					

				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors,reg_loss_mask,reg_targets,result['loc'])
					loss_num += 1
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask],reg_targets[reg_loss_mask]) 
				loss_num += 1

		if self.loss_scale is not None:
			if len(self.loss_scale)==4:
				loss = self.loss_scale[0]*loss_cls + self.loss_scale[1]*loss_loc_1 + self.loss_scale[2]*loss_loc_2 + self.loss_scale[3]*loss_motion
			elif len(self.loss_scale)==3:
				loss = self.loss_scale[0]*loss_cls + self.loss_scale[1]*loss_loc_1 + self.loss_scale[2]*loss_loc_2
			else:
				loss = self.loss_scale[0]*loss_cls + self.loss_scale[1]*loss_loc
		elif not motion_labels is None:
			loss = loss_cls + loss_loc + loss_motion
		else:
			loss = loss_cls + loss_loc

		if loss_num == 2:
			return (loss_num,loss, loss_cls,loss_loc)
		elif loss_num == 3:
			return (loss_num,loss, loss_cls,loss_loc_1,loss_loc_2)
		elif loss_num == 4:
			return (loss_num,loss, loss_cls,loss_loc_1,loss_loc_2,loss_motion)


	def step(self,data,batch_size):
		bev_seq = data['bev_seq']
		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		vis_maps = data['vis_maps']
		trans_matrices = data['trans_matrices']
		num_agent = data['num_agent']

		if self.MGDA:
			self.loss_scale = self.cal_loss_scale(data)
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, trans_matrices, num_agent, vis=None,batch_size=batch_size)

		labels = labels.view(result['cls'].shape[0],-1,result['cls'].shape[-1])
		N = bev_seq.shape[0]


		loss_collect = self.loss_calculator(result,anchors,reg_loss_mask,reg_targets,labels,N)
		loss_num = loss_collect[0]
		if loss_num == 3:
			loss_num,loss, loss_cls,loss_loc_1,loss_loc_2 = loss_collect
		elif loss_num ==2:
			loss_num,loss, loss_cls,loss_loc = loss_collect
		elif loss_num == 4:
			loss_num,loss, loss_cls,loss_loc_1,loss_loc_2,loss_motion = loss_collect

		if self.MGDA:
			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()
			loss.backward()
			self.optimizer_encoder.step()
			self.optimizer_head.step()
		else:

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		if self.config.pred_type in ['motion','center'] and not self.only_det:
			if self.config.motion_state:
				return loss.item(),loss_cls.item(),loss_loc_1.item(),loss_loc_2.item(), loss_motion.item()
			else:
				return loss.item(),loss_cls.item(),loss_loc_1.item(),loss_loc_2.item()
		else:
			return loss.item(),loss_cls.item(),loss_loc.item()

	def predict(self,data,validation=True):

		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq,vis=vis_maps)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0],-1,result['cls'].shape[-1])
			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0],-1,result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result,anchors,reg_loss_mask,reg_targets,labels,N,motion_labels,motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num,loss, loss_cls,loss_loc_1,loss_loc_2 = loss_collect
			elif loss_num ==2:
				loss_num,loss, loss_cls,loss_loc = loss_collect
			elif loss_num == 4:
				loss_num,loss, loss_cls,loss_loc_1,loss_loc_2,loss_motion = loss_collect

			batch_box_preds = result['loc']
			batch_cls_preds = result['cls']

			if self.config.motion_state:
				batch_motion_preds = result['state']
			else:
				batch_motion_preds = None

			if not self.only_det:
				if self.config.pred_type == 'center':
					batch_box_preds[:,:,:,:,1:,2:] = batch_box_preds[:,:,:,:,[0],2:]
 
		class_selected = apply_nms_det(batch_box_preds, batch_cls_preds,anchors,self.code_type,self.config,batch_motion_preds)
		#class_selected = None
		if validation:
			if self.config.pred_type in ['motion','center'] and not self.only_det:
				if self.config.motion_state:
					return loss.item(),loss_cls.item(),loss_loc_1.item(),loss_loc_2.item(), loss_motion.item(),class_selected
				else:
					return loss.item(),loss_cls.item(),loss_loc_1.item(),loss_loc_2.item(),class_selected
			else:
				return loss.item(),loss_cls.item(),loss_loc.item(),class_selected
		else:
			return class_selected

	def predict_all(self,data ,num_sensor,validation=True):
		NUM_AGENT = 5
		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']
		trans_matrices = data['trans_matrices']
		num_agent = data['num_agent']


		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, trans_matrices,  num_agent, vis=None, training=False, inference='activated', batch_size=1)
        
		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0],-1,result['cls'].shape[-1])

			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0],-1,result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result,anchors,reg_loss_mask,reg_targets,labels,N,motion_labels,motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num,loss, loss_cls,loss_loc_1,loss_loc_2 = loss_collect
			elif loss_num ==2:
				loss_num,loss, loss_cls,loss_loc = loss_collect
			elif loss_num == 4:
				loss_num,loss, loss_cls,loss_loc_1,loss_loc_2,loss_motion = loss_collect

		seq_results = [[] for i in range(NUM_AGENT)]
		global_points = [[] for i in range(num_sensor)]
		cls_preds = [[] for i in range(num_sensor)]

		for k in range(NUM_AGENT):	
			bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

			if torch.nonzero(bev_seq).shape[0] == 0:
				seq_results[k] = []
			else:
				batch_box_preds = torch.unsqueeze(result['loc'][k, :, :, :, :, :],0)
				batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
				anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :],0)

				if self.config.motion_state:
					batch_motion_preds = result['state']
				else:
					batch_motion_preds = None

				if not self.only_det:
					if self.config.pred_type == 'center':
						batch_box_preds[:,:,:,:,1:,2:] = batch_box_preds[:,:,:,:,[0],2:]

				class_selected = apply_nms_det(batch_box_preds, batch_cls_preds,anchors,self.code_type,self.config,batch_motion_preds) 
				seq_results[k] = class_selected

		if validation:
			return loss.item(),loss_cls.item(),loss_loc.item(),seq_results
		else:
			return seq_results

	def predict_all_old(self, data, trans_matrices_map, num_sensor, validation=True):
		NUM_AGENT = 5
		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, vis=vis_maps, training=False)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])

			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		seq_results = [[] for i in range(NUM_AGENT)]
		global_points = [[] for i in range(num_sensor)]
		cls_preds = [[] for i in range(num_sensor)]

		for k in range(NUM_AGENT):
			bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

			if torch.nonzero(bev_seq).shape[0] == 0:
				seq_results[k] = []
			else:
				batch_box_preds = torch.unsqueeze(result['loc'][k, :, :, :, :, :], 0)
				batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
				anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)

				if self.config.motion_state:
					batch_motion_preds = result['state']
				else:
					batch_motion_preds = None

				if not self.only_det:
					if self.config.pred_type == 'center':
						batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

				class_selected = apply_nms_det(batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config,
											   batch_motion_preds)
				seq_results[k] = class_selected

				global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k], batch_box_preds,
																			batch_cls_preds, anchors, self.code_type,
																			self.config, batch_motion_preds)

		all_points_scene = numpy.concatenate(tuple(global_points), 0)
		cls_preds_scene = torch.cat(tuple(cls_preds), 0)
		class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

		if validation:
			if self.config.pred_type in ['motion', 'center'] and not self.only_det:
				if self.config.motion_state:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item(), seq_results, class_selected_global
				else:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), seq_results, class_selected_global
			else:
				return loss.item(), loss_cls.item(), loss_loc.item(), seq_results, class_selected_global
		else:
			return seq_results


	def cal_loss_scale(self,data):
		bev_seq = data['bev_seq']
		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		motion_labels = None
		motion_mask = None
		
		with torch.no_grad():
			shared_feats = self.encoder(bev_seq)
		shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
		result = self.head(shared_feats_tensor)
		if self.config.motion_state:
			motion_labels = data['motion_label']
			motion_mask = data['motion_mask']
			motion_labels = motion_labels.view(result['state'].shape[0],-1,result['state'].shape[-1])
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()
		grads = {}
		labels = labels.view(result['cls'].shape[0],-1,result['cls'].shape[-1])
		N = bev_seq.shape[0]

		# calculate loss
		grad_len = 0

		'''
		Classification Loss
		'''
		loss_cls = self.alpha*torch.sum(self.criterion['cls'](result['cls'],labels)) /N
		#loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()

		loss_cls.backward(retain_graph=True)
		grads[0] = []
		grads[0].append(shared_feats_tensor.grad.data.clone().detach())
		shared_feats_tensor.grad.data.zero_()
		grad_len += 1


		'''
		Localization Loss
		'''
		loc_scale = False
		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1,reg_loss_mask.shape[-1])).size(0)


		if self.code_type in ['corner_1','corner_2','corner_3']:
				target = reg_targets[reg_loss_mask].reshape(-1,5,2)
				flip_target = torch.stack([target[:,0],target[:,3],target[:,4],target[:,1],target[:,2]],dim=-2)
				pred = result['loc'][reg_loss_mask].reshape(-1,5,2)
				t = torch.sum(torch.norm(pred-target,dim=-1),dim=-1)
				f = torch.sum(torch.norm(pred-flip_target,dim=-1),dim=-1)
				loss_loc = torch.sum(torch.min(t,f)) / N

		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors,reg_loss_mask,reg_targets,result['loc'])
				elif self.config.pred_type in ['motion','center']:

					###only center/motion for pred
					
					loss_loc_1 = self.corner_loss(anchors,reg_loss_mask[...,0][...,[0]],reg_targets[...,[0],:],result['loc'][...,[0],:])
					pred_reg_loss_mask = reg_loss_mask[...,1:,:]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask #mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][...,1:,:][pred_reg_loss_mask],reg_targets[...,1:,:][pred_reg_loss_mask]) 
					
					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_1.backward(retain_graph=True)
					grads[1] = []
					grads[1].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()		

					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_2.backward(retain_graph=True)
					grads[2] = []
					grads[2].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()	
					loc_scale = True	
					grad_len += 2

				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors,reg_loss_mask,reg_targets,result['loc'])
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask],reg_targets[reg_loss_mask]) 

			if not loc_scale:
				grad_len += 1
				self.optimizer_encoder.zero_grad()
				self.optimizer_head.zero_grad()
				loss_loc.backward(retain_graph=True)
				grads[1] = []
				grads[1].append(shared_feats_tensor.grad.data.clone().detach())
				shared_feats_tensor.grad.data.zero_()	

		'''
		Motion state Loss
		'''
		if self.config.motion_state:
			loss_motion = torch.sum(self.criterion['cls'](result['state'],motion_labels)) /N


			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()

			loss_motion.backward(retain_graph=True)
			grads[3] = []
			grads[3].append(shared_feats_tensor.grad.data.clone().detach())
			shared_feats_tensor.grad.data.zero_()
			grad_len += 1

		# ---------------------------------------------------------------------
		# -- Frank-Wolfe iteration to compute scales.
		scale = np.zeros(grad_len, dtype=np.float32)
		sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(grad_len)])
		for i in range(grad_len):
			scale[i] = float(sol[i])

		#print(scale)
		return scale

class FaFModuleConsensus(object):
	def __init__(self, model, config, optimizer, criterion):
		self.MGDA = config.MGDA
		if self.MGDA:
			self.encoder = model[0]
			self.head = model[1]
			self.optimizer_encoder = optimizer[0]
			self.optimizer_head = optimizer[1]
			self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_encoder,
																		  milestones=[50, 100, 150, 200], gamma=0.5)
			self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_head,
																	   milestones=[50, 100, 150, 200], gamma=0.5)
			self.MGDA = config.MGDA
		else:
			self.model = model
			self.optimizer = optimizer
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
		self.criterion = criterion  # {'cls_loss','loc_loss'}

		self.out_seq_len = config.pred_len
		self.category_num = config.category_num
		self.code_size = config.box_code_size
		self.loss_scale = None

		self.code_type = config.code_type
		self.loss_type = config.loss_type
		self.pred_len = config.pred_len
		self.only_det = config.only_det
		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			self.alpha = 1.
		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				self.alpha = 1.
				if not self.only_det:
					self.alpha = 1.
			else:
				self.alpha = 0.1
		self.config = config

	def resume(self, path):
		def map_func(storage, location):
			return storage.cuda()

		if os.path.isfile(path):
			if rank == 0:
				print("=> loading checkpoint '{}'".format(path))

			checkpoint = torch.load(path, map_location=map_func)
			self.model.load_state_dict(checkpoint['state_dict'], strict=False)

			ckpt_keys = set(checkpoint['state_dict'].keys())
			own_keys = set(model.state_dict().keys())
			missing_keys = own_keys - ckpt_keys
			for k in missing_keys:
				print('caution: missing keys from checkpoint {}: {}'.format(path, k))
		else:
			print("=> no checkpoint found at '{}'".format(path))

	def corner_loss(self, anchors, reg_loss_mask, reg_targets, pred_result,
					indices_association, indices_shape, trans_matrices, trans_matrices_map, target_agent_ids, num_sensor, batch_size):

		device = reg_loss_mask.device

		loss_consensus_loc_total = 0
		for b in range(batch_size):
			for agent_idx in range(5):
				num_consensus_loc_loss = num_sensor[b, agent_idx]
				if num_consensus_loc_loss == 0:
					continue

				target_agent_id = target_agent_ids[b, agent_idx]  # [1], a tensor
				assert target_agent_id == agent_idx, 'target_agent_id does not match the current agent.'

				for nb_agent_id in range(num_consensus_loc_loss):
					if not nb_agent_id == target_agent_id:

						#print(indices_association[target_agent_id, b, nb_agent_id])

						reg_loss_mask_tmp = reg_loss_mask[batch_size * target_agent_id:batch_size * (target_agent_id + 1)]

						pred_result_target = pred_result[batch_size * target_agent_id:batch_size * (target_agent_id + 1)]
						gt_result_target = reg_targets[batch_size * target_agent_id:batch_size * (target_agent_id + 1)]

						pred_result_nb = pred_result[batch_size * nb_agent_id:batch_size * (nb_agent_id + 1)]

						anchors_map_target = anchors[batch_size * target_agent_id:batch_size * (target_agent_id + 1)]
						anchors_map_nb = anchors[batch_size * nb_agent_id:batch_size * (nb_agent_id + 1)]


						mask_indices = reg_loss_mask_tmp[b, :][
										   indices_association[target_agent_id, b, nb_agent_id][0, :indices_shape[target_agent_id, b, nb_agent_id]],
										   indices_association[target_agent_id, b, nb_agent_id][1, :indices_shape[target_agent_id, b, nb_agent_id]], 0, 0] ==True

						mask_consensus_loc_loss = indices_association[target_agent_id, b, nb_agent_id][:,
												  :indices_shape[target_agent_id, b, nb_agent_id]][:, mask_indices]

						if mask_consensus_loc_loss.shape[1] > 0:

							pred_target = pred_result_target[b, :][
								mask_consensus_loc_loss[0, :],
								mask_consensus_loc_loss[1, :]]

							pred_nb = pred_result_nb[b, :][
								mask_consensus_loc_loss[2, :],
								mask_consensus_loc_loss[3, :]]

							anchor_target = anchors_map_target[b, :][
								mask_consensus_loc_loss[0, :],
								mask_consensus_loc_loss[1, :]]

							anchor_nb = anchors_map_nb[b, :][
								mask_consensus_loc_loss[2, :],
								mask_consensus_loc_loss[3, :]]

							gt_target = gt_result_target[b, :][
								mask_consensus_loc_loss[0, :],
								mask_consensus_loc_loss[1, :]]

							gt_decode_target = bev_box_decode_torch(gt_target.view(-1, 6), anchor_target.view(-1, 6))
							gt_corners_target = center_to_corner_box2d_torch(gt_decode_target[..., :2], gt_decode_target[..., 2:4],
																			   gt_decode_target[..., 4:])



							pred_decode_target = bev_box_decode_torch(pred_target.view(-1, 6), anchor_target.view(-1, 6))
							pred_corners_target = center_to_corner_box2d_torch(pred_decode_target[..., :2], pred_decode_target[..., 2:4],
																			   pred_decode_target[..., 4:])

							pred_decode_nb = bev_box_decode_torch(pred_nb.view(-1, 6), anchor_nb.view(-1, 6))
							pred_corners_nb = center_to_corner_box2d_torch(pred_decode_nb[..., :2], pred_decode_nb[..., 2:4],
																			  pred_decode_nb[..., 4:])


							# pred_corners_target = pred_corners_target.cpu().detach().numpy()
							# pred_corners_nb = pred_corners_nb.cpu().detach().numpy()
							# #
							# temp_nb = pred_corners_nb.reshape(-1, 2)
							# local_points_nb = temp_nb.T
							# local_points_nb[0, :] = - local_points_nb[0, :]
							# trans_matrices_map_nb = torch.squeeze(trans_matrices_map[nb_agent_id])
							# trans_matrices_map_nb = trans_matrices_map_nb.cpu().detach().numpy()
							# global_points_nb = np.dot(trans_matrices_map_nb, np.vstack(
							# 	(local_points_nb, np.ones(local_points_nb.shape[1]), np.ones(local_points_nb.shape[1]))))[:2, :]
							# # print(global_points.shape)
							# global_points_nb[0, :] = - global_points_nb[0, :]
							# global_points_nb = global_points_nb.T
							# global_points_nb = global_points_nb.reshape(-1, 4, 2)
							#
							# temp_target = pred_corners_target.reshape(-1, 2)
							# local_points_target = temp_target.T
							# local_points_target[0, :] = - local_points_target[0, :]
							# trans_matrices_map_target = torch.squeeze(trans_matrices_map[target_agent_id])
							# trans_matrices_map_target = trans_matrices_map_target.cpu().detach().numpy()
							# global_points_target = np.dot(trans_matrices_map_target, np.vstack(
							# 	(local_points_target, np.ones(local_points_target.shape[1]), np.ones(local_points_target.shape[1]))))[:2, :]
							# # print(global_points.shape)
							# global_points_target[0, :] = - global_points_target[0, :]
							# global_points_target = global_points_target.T
							# global_points_target = global_points_target.reshape(-1, 4, 2)
							#
							# plt.scatter(np.mean(global_points_nb[:, :, 0], axis=1),
							# 			np.mean(global_points_nb[:, :, 1], axis=1), s=0.2, c='g', alpha=1)
							#
							# plt.scatter(np.mean(global_points_target[:, :, 0], axis=1),
							# 			np.mean(global_points_target[:, :, 1], axis=1),  s=0.2, c='r', alpha=1)
							# plt.show()
							#

	                        ##########################################################################################################
							#print(target_agent_id, nb_agent_id)
							temp = pred_corners_nb.reshape(-1, 2)
							local_points = temp.T
							local_points[0, :] = - local_points[0, :]

							trans_matrices_nb_to_target = trans_matrices[target_agent_id, b, nb_agent_id].cpu().detach().numpy()

							trans_matrices_nb_to_target[:3, 3] = trans_matrices_nb_to_target[:3, 3]/4

							global_points = np.dot(trans_matrices_nb_to_target, np.vstack(
								(local_points.cpu().detach().numpy(), np.ones(local_points.shape[1]), np.ones(local_points.shape[1]))))[:2, :]
							# print(global_points.shape)
							global_points[0, :] = - global_points[0, :]
							global_points = global_points.T
							global_points = global_points.reshape(-1, 4, 2)

							pred_corners_nb_to_target = torch.Tensor(global_points).to(device)

							#print(pred_corners_nb_to_target.shape, pred_corners_target.shape)
							loss_consensus_loc_total = loss_consensus_loc_total +\
													   torch.sum(torch.norm(pred_corners_target - pred_corners_nb_to_target, dim=-1))
							#print(loss_consensus_loc_total)

							pred_corners_target = pred_corners_target.cpu().detach().numpy()
							pred_corners_nb_to_target = pred_corners_nb_to_target.cpu().detach().numpy()

							gt_corners_target = gt_corners_target.cpu().detach().numpy()

							plt.scatter(np.mean(pred_corners_target[:, :, 0], axis=1),
							np.mean(pred_corners_target[:, :, 1], axis=1), s=0.2, c='g', alpha=0.2)

							plt.scatter(np.mean(gt_corners_target[:, :, 0], axis=1),
							np.mean(gt_corners_target[:, :, 1], axis=1), s=35, c='b' ,marker='*', alpha=1)

							plt.scatter(np.mean(pred_corners_nb_to_target[:, :, 0], axis=1),
							np.mean(pred_corners_nb_to_target[:, :, 1], axis=1),  s=0.2, c='r', alpha=0.2)
							plt.show()


		N = pred_result.shape[0]
		anchors = anchors.unsqueeze(-2).expand(anchors.shape[0], anchors.shape[1], anchors.shape[2], anchors.shape[3],
											   reg_loss_mask.shape[-1], anchors.shape[-1])
		assigned_anchor = anchors[reg_loss_mask]
		assigned_target = reg_targets[reg_loss_mask]
		assigned_pred = pred_result[reg_loss_mask]

		pred_decode = bev_box_decode_torch(assigned_pred, assigned_anchor)
		target_decode = bev_box_decode_torch(assigned_target, assigned_anchor)
		pred_corners = center_to_corner_box2d_torch(pred_decode[..., :2], pred_decode[..., 2:4], pred_decode[..., 4:])
		target_corners = center_to_corner_box2d_torch(target_decode[..., :2], target_decode[..., 2:4],
													  target_decode[..., 4:])
		loss_loc = torch.sum(torch.norm(pred_corners - target_corners, dim=-1)) / N

		#return loss_loc
		return loss_loc + 0.05*loss_consensus_loc_total/(batch_size*num_sensor[0,0]*(num_sensor[0,0]-1))

	def loss_calculator(self, result, anchors, reg_loss_mask, reg_targets, labels, N,
						indices_association, indices_shape, trans_matrices, trans_matrices_map, target_agent_ids, num_sensor, batch_size, motion_labels=None,
						motion_mask=None):

		# agents_cls = result['cls'].view(N, 256, 256, 6, 2)
		# loss_consensus_cls_total = 0
		#
		# #print(agents_cls.shape, num_sensor.shape, target_agent_ids.shape)
		#
		# for b in range(batch_size):
		# 	for agent_idx in range(5):
		# 		num_consensus_cls_loss = num_sensor[b, agent_idx]
		# 		if num_consensus_cls_loss == 0:
		# 			continue
		#
		# 		target_agent_id = target_agent_ids[b, agent_idx]  # [1], a tensor
		# 		assert target_agent_id == agent_idx, 'target_agent_id does not match the current agent.'
		#
		# 		for nb_agent_id in range(num_consensus_cls_loss):
		# 			if not nb_agent_id == target_agent_id:
		#
		# 				pred_result_target = agents_cls[batch_size * target_agent_id:batch_size * (target_agent_id + 1)]
		# 				pred_result_nb = agents_cls[batch_size * nb_agent_id:batch_size * (nb_agent_id + 1)]
		#
		# 				pred_result_target_overlap = pred_result_target[b, indices_association[target_agent_id, b, nb_agent_id][0, :indices_shape[target_agent_id, b, nb_agent_id]],
		# 				indices_association[target_agent_id, b, nb_agent_id][1, :indices_shape[target_agent_id, b, nb_agent_id]], :, :]
		#
		# 				pred_result_nb_overlap = pred_result_nb[b, indices_association[target_agent_id, b, nb_agent_id][2, :indices_shape[target_agent_id, b, nb_agent_id]],
		# 				indices_association[target_agent_id, b, nb_agent_id][3, :indices_shape[target_agent_id, b, nb_agent_id]], :, :]
		#
		# 				if pred_result_target_overlap.shape[0] > 1000:
		#
		# 					pred_result_target_overlap = pred_result_target_overlap.view(-1, 2)
		# 					pred_result_nb_overlap = pred_result_nb_overlap.view(-1, 2)
		#
		# 					pred_result_target_overlap_probabilities = F.softmax(pred_result_target_overlap, dim=-1)
		# 					pred_result_nb_overlap_probabilities = F.softmax(pred_result_nb_overlap, dim=-1)
		#
		# 					#print(pred_result_target_overlap_probabilities, pred_result_nb_overlap_probabilities)
		#
		# 					loss_consensus_cls_total = loss_consensus_cls_total + \
		# 											   torch.sum(torch.norm(pred_result_target_overlap_probabilities - pred_result_nb_overlap_probabilities, p=1, dim=-1))

		loss_num = 0
		# calculate loss
		weights = torch.Tensor([0.005, 1.0, 1.0, 1.0, 1.0]).cuda().double()
		loss_cls = torch.sum(self.criterion['cls'](result['cls'], labels)) / N #+ 0.01*loss_consensus_cls_total/(batch_size*num_sensor[0,0]*(num_sensor[0,0]-1))


		loss_num += 1
		# loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N

		# Motion state
		if not motion_labels is None:
			loss_motion = torch.sum(self.criterion['cls'](result['state'], motion_labels)) / N
			loss_num += 1

		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)
		# print(loss_mask_num)
		# print(torch.sum(reg_targets[:,:,:,:,0][reg_loss_mask[:,:,:,:,2]]))

		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
			flip_target = torch.stack([target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
			pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
			t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
			f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
			loss_loc = torch.sum(torch.min(t, f)) / N


		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'],
												indices_association, indices_shape, trans_matrices, trans_matrices_map, target_agent_ids, num_sensor, batch_size)
					loss_num += 1
				elif self.config.pred_type in ['motion', 'center']:

					###only center/motion for pred

					loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [0]], reg_targets[..., [0], :],
												  result['loc'][..., [0], :])
					pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask  # mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][..., 1:, :][pred_reg_loss_mask],
												  reg_targets[..., 1:, :][pred_reg_loss_mask])
					loss_loc = loss_loc_1 + loss_loc_2
					loss_num += 2


				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
					loss_num += 1
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])
				loss_num += 1

		if self.loss_scale is not None:
			if len(self.loss_scale) == 4:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc_1 + self.loss_scale[
					2] * loss_loc_2 + self.loss_scale[3] * loss_motion
			elif len(self.loss_scale) == 3:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc_1 + self.loss_scale[2] * loss_loc_2
			else:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc
		elif not motion_labels is None:
			loss = loss_cls + loss_loc + loss_motion
		else:
			loss = loss_cls + loss_loc

		if loss_num == 2:
			return (loss_num, loss, loss_cls, loss_loc)
		elif loss_num == 3:
			return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2)
		elif loss_num == 4:
			return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion)

	def step(self, data, batch_size, trans_matrices_map):
		bev_seq = data['bev_seq']
		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		vis_maps = data['vis_maps']
		motion_labels = None
		motion_mask = None

		target_agent_ids = data['target_agent_ids']
		num_agent = data['num_agent']
		indices_association = data['indices_association']
		indices_shape = data['indices_shape']
		trans_matrices = data['trans_matrices']

		if self.MGDA:
			self.loss_scale = self.cal_loss_scale(data)
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, vis=vis_maps, batch_size=batch_size)

		labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])

		if self.config.motion_state:
			motion_labels = data['motion_label']
			motion_mask = data['motion_mask']
			motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
		N = bev_seq.shape[0]

		loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N,
											indices_association, indices_shape, trans_matrices, trans_matrices_map, target_agent_ids, num_agent, batch_size, motion_labels, motion_mask)

		loss_num = loss_collect[0]
		if loss_num == 3:
			loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
		elif loss_num == 2:
			loss_num, loss, loss_cls, loss_loc = loss_collect
		elif loss_num == 4:
			loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		if self.MGDA:
			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()
			loss.backward()
			self.optimizer_encoder.step()
			self.optimizer_head.step()
		else:

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		if self.config.pred_type in ['motion', 'center'] and not self.only_det:
			if self.config.motion_state:
				return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item()
			else:
				return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item()
		else:
			return loss.item(), loss_cls.item(), loss_loc.item()

	def predict(self, data, validation=True):

		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, vis=vis_maps)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

			batch_box_preds = result['loc']
			batch_cls_preds = result['cls']

			if self.config.motion_state:
				batch_motion_preds = result['state']
			else:
				batch_motion_preds = None

			if not self.only_det:
				if self.config.pred_type == 'center':
					batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

		class_selected = apply_nms_det(batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config,
									   batch_motion_preds)
		# class_selected = None
		if validation:
			if self.config.pred_type in ['motion', 'center'] and not self.only_det:
				if self.config.motion_state:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item(), class_selected
				else:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), class_selected
			else:
				return loss.item(), loss_cls.item(), loss_loc.item(), class_selected
		else:
			return class_selected

	def predict_all(self, data, trans_matrices_map, num_sensor, validation=True):
		NUM_AGENT = 5
		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, vis=vis_maps, training=False)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])

			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		seq_results = [[] for i in range(NUM_AGENT)]
		global_points = [[] for i in range(num_sensor)]
		cls_preds = [[] for i in range(num_sensor)]

		for k in range(NUM_AGENT):
			bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

			if torch.nonzero(bev_seq).shape[0] == 0:
				seq_results[k] = []
			else:
				batch_box_preds = torch.unsqueeze(result['loc'][k, :, :, :, :, :], 0)
				batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
				anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)

				if self.config.motion_state:
					batch_motion_preds = result['state']
				else:
					batch_motion_preds = None

				if not self.only_det:
					if self.config.pred_type == 'center':
						batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

				class_selected = apply_nms_det(batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config,
											   batch_motion_preds)
				seq_results[k] = class_selected

				global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k], batch_box_preds,
																			batch_cls_preds, anchors, self.code_type,
																			self.config, batch_motion_preds)

		all_points_scene = numpy.concatenate(tuple(global_points), 0)
		cls_preds_scene = torch.cat(tuple(cls_preds), 0)
		class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

		if validation:
			if self.config.pred_type in ['motion', 'center'] and not self.only_det:
				if self.config.motion_state:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item(), seq_results, class_selected_global
				else:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), seq_results, class_selected_global
			else:
				return loss.item(), loss_cls.item(), loss_loc.item(), seq_results, class_selected_global
		else:
			return seq_results

	def cal_loss_scale(self, data):
		bev_seq = data['bev_seq']
		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		motion_labels = None
		motion_mask = None

		with torch.no_grad():
			shared_feats = self.encoder(bev_seq)
		shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
		result = self.head(shared_feats_tensor)
		if self.config.motion_state:
			motion_labels = data['motion_label']
			motion_mask = data['motion_mask']
			motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()
		grads = {}
		labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
		N = bev_seq.shape[0]

		# calculate loss
		grad_len = 0

		'''
		Classification Loss
		'''
		loss_cls = self.alpha * torch.sum(self.criterion['cls'](result['cls'], labels)) / N
		# loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()

		loss_cls.backward(retain_graph=True)
		grads[0] = []
		grads[0].append(shared_feats_tensor.grad.data.clone().detach())
		shared_feats_tensor.grad.data.zero_()
		grad_len += 1

		'''
		Localization Loss
		'''
		loc_scale = False
		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)

		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
			flip_target = torch.stack([target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
			pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
			t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
			f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
			loss_loc = torch.sum(torch.min(t, f)) / N

		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
				elif self.config.pred_type in ['motion', 'center']:

					###only center/motion for pred

					loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [0]], reg_targets[..., [0], :],
												  result['loc'][..., [0], :])
					pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask  # mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][..., 1:, :][pred_reg_loss_mask],
												  reg_targets[..., 1:, :][pred_reg_loss_mask])

					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_1.backward(retain_graph=True)
					grads[1] = []
					grads[1].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()

					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_2.backward(retain_graph=True)
					grads[2] = []
					grads[2].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()
					loc_scale = True
					grad_len += 2

				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])

			if not loc_scale:
				grad_len += 1
				self.optimizer_encoder.zero_grad()
				self.optimizer_head.zero_grad()
				loss_loc.backward(retain_graph=True)
				grads[1] = []
				grads[1].append(shared_feats_tensor.grad.data.clone().detach())
				shared_feats_tensor.grad.data.zero_()

		'''
		Motion state Loss
		'''
		if self.config.motion_state:
			loss_motion = torch.sum(self.criterion['cls'](result['state'], motion_labels)) / N

			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()

			loss_motion.backward(retain_graph=True)
			grads[3] = []
			grads[3].append(shared_feats_tensor.grad.data.clone().detach())
			shared_feats_tensor.grad.data.zero_()
			grad_len += 1

		# ---------------------------------------------------------------------
		# -- Frank-Wolfe iteration to compute scales.
		scale = np.zeros(grad_len, dtype=np.float32)
		sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(grad_len)])
		for i in range(grad_len):
			scale[i] = float(sol[i])

		# print(scale)
		return scale


class FaFModuleKD(object):
	def __init__(self, model, teacher, config, optimizer, criterion):
		self.MGDA = config.MGDA
		if self.MGDA:
			self.encoder = model[0]
			self.head = model[1]
			self.optimizer_encoder = optimizer[0]
			self.optimizer_head = optimizer[1]
			self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_encoder, milestones=[50, 100, 150, 200], gamma=0.5)
			self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_head, milestones=[50, 100, 150, 200], gamma=0.5)
			self.MGDA = config.MGDA
		else:
			self.model = model
			self.teacher = teacher
			for k, v in self.teacher.named_parameters():
				v.requires_grad = False  # fix parameters
			self.optimizer = optimizer
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
		self.criterion = criterion  # {'cls_loss','loc_loss'}
		self.out_seq_len = config.pred_len
		self.category_num = config.category_num
		self.code_size = config.box_code_size
		self.loss_scale = None

		self.code_type = config.code_type
		self.loss_type = config.loss_type
		self.pred_len = config.pred_len
		self.only_det = config.only_det
		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			self.alpha = 1.
		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				self.alpha = 1.
				if not self.only_det:
					self.alpha = 1.
			else:
				self.alpha = 0.1
		self.config = config

	def resume(self, path):
		def map_func(storage, location):
			return storage.cuda()

		if os.path.isfile(path):
			if rank == 0:
				print("=> loading checkpoint '{}'".format(path))

			checkpoint = torch.load(path, map_location=map_func)
			self.model.load_state_dict(checkpoint['state_dict'], strict=False)

			ckpt_keys = set(checkpoint['state_dict'].keys())
			own_keys = set(model.state_dict().keys())
			missing_keys = own_keys - ckpt_keys
			for k in missing_keys:
				print('caution: missing keys from checkpoint {}: {}'.format(path, k))
		else:
			print("=> no checkpoint found at '{}'".format(path))

	def corner_loss(self, anchors, reg_loss_mask, reg_targets, pred_result):
		N = pred_result.shape[0]
		anchors = anchors.unsqueeze(-2).expand(anchors.shape[0], anchors.shape[1], anchors.shape[2], anchors.shape[3],
											   reg_loss_mask.shape[-1], anchors.shape[-1])
		assigned_anchor = anchors[reg_loss_mask]
		assigned_target = reg_targets[reg_loss_mask]
		assigned_pred = pred_result[reg_loss_mask]
		# print(assigned_anchor.shape,assigned_pred.shape,assigned_target.shape)
		# exit()
		pred_decode = bev_box_decode_torch(assigned_pred, assigned_anchor)
		target_decode = bev_box_decode_torch(assigned_target, assigned_anchor)
		pred_corners = center_to_corner_box2d_torch(pred_decode[..., :2], pred_decode[..., 2:4], pred_decode[..., 4:])
		target_corners = center_to_corner_box2d_torch(target_decode[..., :2], target_decode[..., 2:4],
													  target_decode[..., 4:])
		loss_loc = torch.sum(torch.norm(pred_corners - target_corners, dim=-1)) / N

		return loss_loc

	def loss_calculator(self, result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels=None,
						motion_mask=None):
		loss_num = 0
		# calculate loss
		weights = torch.Tensor([0.005, 1.0, 1.0, 1.0, 1.0]).cuda().double()
		loss_cls = torch.sum(self.criterion['cls'](result['cls'], labels)) / N
		loss_num += 1
		# loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N

		# Motion state
		if not motion_labels is None:
			loss_motion = torch.sum(self.criterion['cls'](result['state'], motion_labels)) / N
			loss_num += 1

		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)
		# print(loss_mask_num)
		# print(torch.sum(reg_targets[:,:,:,:,0][reg_loss_mask[:,:,:,:,2]]))

		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
			flip_target = torch.stack([target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
			pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
			t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
			f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
			loss_loc = torch.sum(torch.min(t, f)) / N


		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
					loss_num += 1
				elif self.config.pred_type in ['motion', 'center']:

					###only center/motion for pred

					loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [0]], reg_targets[..., [0], :],
												  result['loc'][..., [0], :])
					pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask  # mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][..., 1:, :][pred_reg_loss_mask],
												  reg_targets[..., 1:, :][pred_reg_loss_mask])
					loss_loc = loss_loc_1 + loss_loc_2
					loss_num += 2


				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
					loss_num += 1
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])
				loss_num += 1

		if self.loss_scale is not None:
			if len(self.loss_scale) == 4:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc_1 + self.loss_scale[
					2] * loss_loc_2 + self.loss_scale[3] * loss_motion
			elif len(self.loss_scale) == 3:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc_1 + self.loss_scale[2] * loss_loc_2
			else:
				loss = self.loss_scale[0] * loss_cls + self.loss_scale[1] * loss_loc
		elif not motion_labels is None:
			loss = loss_cls + loss_loc + loss_motion
		else:
			loss = loss_cls + loss_loc

		if loss_num == 2:
			return (loss_num, loss, loss_cls, loss_loc)
		elif loss_num == 3:
			return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2)
		elif loss_num == 4:
			return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion)

	def step(self, data, batch_size):
		bev_seq = data['bev_seq']
		bev_seq_teacher = data['bev_seq_teacher']
		kd_weight = data['kd_weight']
		# kd_loss_type = data['kd_loss_type']
		layer = data['layer']

		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		vis_maps = data['vis_maps']
		trans_matrices = data['trans_matrices']
		num_agent = data['num_agent']

		# with torch.autograd.set_detect_anomaly(True):
		if self.MGDA:
			self.loss_scale = self.cal_loss_scale(data)
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result, x_8, x_7, x_6, x_5, fused_layer = self.model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)
			# result, x_8, x_7, x_6, x_5, fused_layer, _ = self.model(bev_seq)

		x_8_teacher, x_7_teacher, x_6_teacher, x_5_teacher, x_3_teacher, x_2_teacher = self.teacher(bev_seq_teacher, vis=vis_maps)

		# for k, v in self.teacher.named_parameters():
		# 	if k != 'xxx.weight' and k != 'xxx.bias':
		# 		print(v.requires_grad)  # 理想状态下，所有值都是False

		# for k, v in self.model.named_parameters():
		# 	if k != 'xxx.weight' and k != 'xxx.bias':
		# 		print(v.requires_grad)  # 理想状态下，所有值都是False

		# -------- KD loss---------#
		kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

		target_x8 = x_8_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
		student_x8 = x_8.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
		kd_loss_x8 = kl_loss_mean(F.log_softmax(student_x8, dim=1), F.softmax(target_x8, dim=1))
		# #
		target_x7 = x_7_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 128 * 128, -1)
		student_x7 = x_7.permute(0, 2, 3, 1).reshape(5 * batch_size * 128 * 128, -1)
		kd_loss_x7 = kl_loss_mean(F.log_softmax(student_x7, dim=1), F.softmax(target_x7, dim=1))
		#
		target_x6 = x_6_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 64 * 64, -1)
		student_x6 = x_6.permute(0, 2, 3, 1).reshape(5 * batch_size * 64 * 64, -1)
		kd_loss_x6 = kl_loss_mean(F.log_softmax(student_x6, dim=1), F.softmax(target_x6, dim=1))
		# #
		target_x5 = x_5_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
		student_x5 = x_5.permute(0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
		kd_loss_x5 = kl_loss_mean(F.log_softmax(student_x5, dim=1), F.softmax(target_x5, dim=1))

		if layer == 3:
			target_x3 = x_3_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
			student_x3 = fused_layer.permute(0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
			kd_loss_fused_layer = kl_loss_mean(F.log_softmax(student_x3, dim=1), F.softmax(target_x3, dim=1))
			#kd_loss = kd_weight * (kd_loss_fused_layer)
			# kd_loss = kd_weight * (kd_loss_x5 + kd_loss_fused_layer)
			# kd_loss = kd_weight * (kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
			# kd_loss = kd_weight * (kd_loss_x7 + kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
			kd_loss = kd_weight * (kd_loss_x8 + kd_loss_x7 + kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
		elif layer == 2:
			target_x2 = x_2_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 64 * 64, -1)
			student_x2 = fused_layer.permute(0, 2, 3, 1).reshape(5 * batch_size * 64 * 64, -1)
			kd_loss_fused_layer = kl_loss_mean(F.log_softmax(student_x2, dim=1), F.softmax(target_x2, dim=1))
			kd_loss = kd_weight * (kd_loss_x8 + kd_loss_x7 + kd_loss_x6  + kd_loss_fused_layer)

		print(kd_loss)

		labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
		N = bev_seq.shape[0]
		loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N)


		# -------- for debugging teacher model---------#
		# loss_collect_teacher = self.loss_calculator(result_teacher,anchors,reg_loss_mask,reg_targets,labels,N)
		# loss_num, loss, loss_cls, loss_loc = loss_collect_teacher
		# print(loss, loss_cls, loss_loc)

		loss_num = loss_collect[0]
		if loss_num == 3:
			loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
		elif loss_num == 2:
			loss_num, loss, loss_cls, loss_loc = loss_collect
		elif loss_num == 4:
			loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		loss = loss + kd_loss

		if self.MGDA:
			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()
			loss.backward()
			self.optimizer_encoder.step()
			self.optimizer_head.step()
		else:
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		if self.config.pred_type in ['motion', 'center'] and not self.only_det:
			if self.config.motion_state:
				return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item()
			else:
				return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item()
		else:
			return loss.item(), loss_cls.item(), loss_loc.item()

	def predict(self, data, validation=True):

		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, vis=vis_maps)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

			batch_box_preds = result['loc']
			batch_cls_preds = result['cls']

			if self.config.motion_state:
				batch_motion_preds = result['state']
			else:
				batch_motion_preds = None

			if not self.only_det:
				if self.config.pred_type == 'center':
					batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

		class_selected = apply_nms_det(batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config,
									   batch_motion_preds)
		# class_selected = None
		if validation:
			if self.config.pred_type in ['motion', 'center'] and not self.only_det:
				if self.config.motion_state:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item(), class_selected
				else:
					return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), class_selected
			else:
				return loss.item(), loss_cls.item(), loss_loc.item(), class_selected
		else:
			return class_selected

	def predict_all(self, data, batch_size, validation=True):
		NUM_AGENT = 5
		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']
		trans_matrices = data['trans_matrices']
		num_agent_tensor = data['num_agent']
		num_sensor = num_agent_tensor[0, 0]

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, trans_matrices, num_agent_tensor, batch_size=batch_size)
		# result = self.model(bev_seq,vis=vis_maps,training=False)
		#
		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])

			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		seq_results = [[] for i in range(NUM_AGENT)]

		for k in range(NUM_AGENT):
			bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

			if torch.nonzero(bev_seq).shape[0] == 0:
				seq_results[k] = []
			else:
				batch_box_preds = torch.unsqueeze(result['loc'][k, :, :, :, :, :], 0)
				batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
				anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)
				batch_motion_preds = None

				if not self.only_det:
					if self.config.pred_type == 'center':
						batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

				class_selected, box_scores_pred_cls = apply_nms_det(batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config, batch_motion_preds)
				seq_results[k] = class_selected

		if validation:
			return loss.item(), loss_cls.item(), loss_loc.item(), seq_results
		else:
			return seq_results

	def predict_all_with_box_com(self, data, trans_matrices_map, validation=True):
		NUM_AGENT = 5
		bev_seq = data['bev_seq']
		vis_maps = data['vis_maps']
		trans_matrices = data['trans_matrices']
		num_agent_tensor = data['num_agent']
		num_sensor = num_agent_tensor[0, 0]

		if self.MGDA:
			x = self.encoder(bev_seq)
			result = self.head(x)
		else:
			result = self.model(bev_seq, trans_matrices, num_agent_tensor, batch_size=1)

		N = bev_seq.shape[0]

		if validation:
			labels = data['labels']
			anchors = data['anchors']
			reg_targets = data['reg_targets']
			reg_loss_mask = data['reg_loss_mask']
			motion_labels = None
			motion_mask = None

			labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])

			if self.config.motion_state:
				motion_labels = data['motion_label']
				motion_mask = data['motion_mask']
				motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
			N = bev_seq.shape[0]

			loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
												motion_mask)
			loss_num = loss_collect[0]
			if loss_num == 3:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
			elif loss_num == 2:
				loss_num, loss, loss_cls, loss_loc = loss_collect
			elif loss_num == 4:
				loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

		seq_results = [[] for i in range(NUM_AGENT)]
		local_results_wo_local_nms = [[] for i in range(NUM_AGENT)]
		local_results_af_local_nms = [[] for i in range(NUM_AGENT)]

		global_points = [[] for i in range(num_sensor)]
		cls_preds = [[] for i in range(num_sensor)]
		global_boxes_af_localnms = [[] for i in range(num_sensor)]
		box_scores_af_localnms = [[] for i in range(num_sensor)]

		forward_message_size = 0
		forward_message_size_two_nms = 0

		for k in range(NUM_AGENT):
			bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

			if torch.nonzero(bev_seq).shape[0] == 0:
				seq_results[k] = []
			else:
				batch_box_preds = torch.unsqueeze(result['loc'][k, :, :, :, :, :], 0)
				batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
				anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)

				if self.config.motion_state:
					batch_motion_preds = result['state']
				else:
					batch_motion_preds = None

				if not self.only_det:
					if self.config.pred_type == 'center':
						batch_box_preds[:, :, :, :, 1:, 2:] = batch_box_preds[:, :, :, :, [0], 2:]

				class_selected, box_scores_pred_cls = apply_nms_det(batch_box_preds, batch_cls_preds, anchors,
																	self.code_type, self.config, batch_motion_preds)

				# transform all the boxes before local nms to the global coordinate
				# global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k], batch_box_preds,
				#                                                            batch_cls_preds, anchors, self.code_type,
				#                                                            self.config, batch_motion_preds)

				# transform the boxes after local nms to the global coordinate
				global_boxes_af_localnms[k], box_scores_af_localnms[k] = apply_box_global_transform_af_localnms(
					trans_matrices_map[k], class_selected, box_scores_pred_cls)
				# print(cls_preds[k].shape, box_scores_af_localnms[k].shape)

				forward_message_size = forward_message_size + 256 * 256 * 6 * 4 * 2
				forward_message_size_two_nms = forward_message_size_two_nms + global_boxes_af_localnms[k].shape[
					0] * 4 * 2

		# global results with one NMS
		# all_points_scene = numpy.concatenate(tuple(global_points), 0)
		# cls_preds_scene = torch.cat(tuple(cls_preds), 0)
		# class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

		# global results with two NMS
		global_boxes_af_local_nms = numpy.concatenate(tuple(global_boxes_af_localnms), 0)
		box_scores_af_local_nms = torch.cat(tuple(box_scores_af_localnms), 0)
		class_selected_global_af_local_nms = apply_nms_global_scene(global_boxes_af_local_nms, box_scores_af_local_nms)

		# transform the consensus global boxes to local agents (two NMS)
		back_message_size_two_nms = 0
		for k in range(num_sensor):
			local_results_af_local_nms[k], ms = apply_box_local_transform(class_selected_global_af_local_nms,
																		  trans_matrices_map[k])
			back_message_size_two_nms = back_message_size_two_nms + ms

		sample_bandwidth_two_nms = forward_message_size_two_nms + back_message_size_two_nms

		# transform the consensus global boxes to local agents (One NMS)
		# back_message_size = 0
		# for k in range(num_sensor):
		#    local_results_wo_local_nms[k], ms = apply_box_local_transform(class_selected_global, trans_matrices_map[k])
		#    back_message_size = back_message_size + ms

		# sample_bandwidth = forward_message_size + back_message_size

		return loss.item(), loss_cls.item(), loss_loc.item(), local_results_af_local_nms, class_selected_global_af_local_nms, sample_bandwidth_two_nms

	def cal_loss_scale(self, data):
		bev_seq = data['bev_seq']
		labels = data['labels']
		reg_targets = data['reg_targets']
		reg_loss_mask = data['reg_loss_mask']
		anchors = data['anchors']
		motion_labels = None
		motion_mask = None

		with torch.no_grad():
			shared_feats = self.encoder(bev_seq)
		shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
		result = self.head(shared_feats_tensor)
		if self.config.motion_state:
			motion_labels = data['motion_label']
			motion_mask = data['motion_mask']
			motion_labels = motion_labels.view(result['state'].shape[0], -1, result['state'].shape[-1])
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()
		grads = {}
		labels = labels.view(result['cls'].shape[0], -1, result['cls'].shape[-1])
		N = bev_seq.shape[0]

		# calculate loss
		grad_len = 0

		'''
		Classification Loss
		'''
		loss_cls = self.alpha * torch.sum(self.criterion['cls'](result['cls'], labels)) / N
		# loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N
		self.optimizer_encoder.zero_grad()
		self.optimizer_head.zero_grad()

		loss_cls.backward(retain_graph=True)
		grads[0] = []
		grads[0].append(shared_feats_tensor.grad.data.clone().detach())
		shared_feats_tensor.grad.data.zero_()
		grad_len += 1

		'''
		Localization Loss
		'''
		loc_scale = False
		loss_mask_num = torch.nonzero(reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)

		if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
			target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
			flip_target = torch.stack([target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
			pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
			t = torch.sum(torch.norm(pred - target, dim=-1), dim=-1)
			f = torch.sum(torch.norm(pred - flip_target, dim=-1), dim=-1)
			loss_loc = torch.sum(torch.min(t, f)) / N

		elif self.code_type == 'faf':
			if self.loss_type == 'corner_loss':
				if self.only_det:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
				elif self.config.pred_type in ['motion', 'center']:

					###only center/motion for pred

					loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [0]], reg_targets[..., [0], :],
												  result['loc'][..., [0], :])
					pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
					if self.config.motion_state:
						pred_reg_loss_mask = motion_mask  # mask out static object
					loss_loc_2 = F.smooth_l1_loss(result['loc'][..., 1:, :][pred_reg_loss_mask],
												  reg_targets[..., 1:, :][pred_reg_loss_mask])

					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_1.backward(retain_graph=True)
					grads[1] = []
					grads[1].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()

					self.optimizer_encoder.zero_grad()
					self.optimizer_head.zero_grad()

					loss_loc_2.backward(retain_graph=True)
					grads[2] = []
					grads[2].append(shared_feats_tensor.grad.data.clone().detach())
					shared_feats_tensor.grad.data.zero_()
					loc_scale = True
					grad_len += 2

				###corners for pred
				else:
					loss_loc = self.corner_loss(anchors, reg_loss_mask, reg_targets, result['loc'])
			else:

				loss_loc = F.smooth_l1_loss(result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])

			if not loc_scale:
				grad_len += 1
				self.optimizer_encoder.zero_grad()
				self.optimizer_head.zero_grad()
				loss_loc.backward(retain_graph=True)
				grads[1] = []
				grads[1].append(shared_feats_tensor.grad.data.clone().detach())
				shared_feats_tensor.grad.data.zero_()

		'''
		Motion state Loss
		'''
		if self.config.motion_state:
			loss_motion = torch.sum(self.criterion['cls'](result['state'], motion_labels)) / N

			self.optimizer_encoder.zero_grad()
			self.optimizer_head.zero_grad()

			loss_motion.backward(retain_graph=True)
			grads[3] = []
			grads[3].append(shared_feats_tensor.grad.data.clone().detach())
			shared_feats_tensor.grad.data.zero_()
			grad_len += 1

		# ---------------------------------------------------------------------
		# -- Frank-Wolfe iteration to compute scales.
		scale = np.zeros(grad_len, dtype=np.float32)
		sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(grad_len)])
		for i in range(grad_len):
			scale[i] = float(sol[i])

		# print(scale)
		return scale