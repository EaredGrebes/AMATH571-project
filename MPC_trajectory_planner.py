import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import partial
import time

torch.set_default_dtype(torch.float64)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

#-----------------------------------------------------------------------------------------
# optimizer class
class trajectoryOptimizer():
    def __init__(self, sigma_u, sigma_a, x_dim, u_dim, n_points, A_fun, A_jacobian, device):
        
        self.device = device
        #self.device = 'cpu'
        # cost function parameters
        self.sigma_u = sigma_u   # penalty for w (rate)
        self.sigma_a = sigma_a    # penalty for accel
        
        self.A_fun      = A_fun
        self.A_jacobian = A_jacobian

        # trajectory
        self.n_points = n_points
        self.x_dim    = x_dim
        self.u_dim    = u_dim

        self.nu = self.u_dim * (self.n_points + 1)          # number of w points that define a trajectory profile
        self.nu_opt = self.u_dim * self.n_points  # minus 1 boundary condition points that constrain u at one end
        #self.n_opt = self.nw_opt + 1       # the w points, minus the two boundary points, plus one dt
       
        # indexes of optimization vector
        self.id_u1 = torch.arange(self.nu)  # every point in the uMat
        self.id_u2 = torch.arange(self.u_dim, self.u_dim + self.nu_opt)      # every point in the wMat minus the one boundary condition

        # assigned in optimize_trajectory
        self.sigma0    = torch.zeros(self.u_dim)
        self.sigmaf    = torch.zeros(self.u_dim)
        self.A_k0_tens = torch.zeros((self.x_dim, self.x_dim, self.n_points))
        self.uMat      = torch.zeros((self.u_dim, self.n_points))
        self.aMat      = torch.zeros((self.u_dim, self.n_points))
        self.timeVec   = torch.zeros(self.n_points)
        #self.dt = 2
        #self.slew_duration = 0

        # construct hessian matrix of cost function
        self.sigma_u = self.sigma_u.repeat(self.n_points+1)
        self.sigma_a = self.sigma_a.repeat(self.n_points+1)

        Ha = torch.diag(torch.ones(self.nu)) - 2*torch.diag(torch.ones(self.nu-u_dim),u_dim) - 2*torch.diag(torch.ones(self.nu-u_dim),-u_dim)
        Ha = torch.diag(self.sigma_a) @ Ha
        Hu = torch.diag(self.sigma_u) + Ha

        # remove the first indexes in both axes of the matrix
        # these indexes are the boundary condition rates, they are not optimized 
        Hu = Hu[self.id_u2,:]
        Hu = Hu[:,self.id_u2]
        self.Hinv = torch.linalg.inv(Hu)
        self.I = torch.eye(self.nu_opt)
        
        # temporary matricies for forward pass
        self.A_big = torch.zeros((self.x_dim*self.n_points, self.x_dim*self.n_points), device=self.device)
        
        self.dx_du_tensor  = torch.zeros((self.n_points, self.x_dim, self.u_dim), device = self.device)
        self.dc_dx_vec     = torch.zeros(self.x_dim * self.n_points, device = self.device)
        
        # debug
        eigVals = torch.linalg.eig(Hu)[0].numpy()
        print(f'H min eig val (must be positive) {eigVals.min()}')
        bp=1
        
        # optimizer hyper-parameters
        self.k1 = 0.008
        self.k2 = 0.1
        self.g_norm_lim = 5e-4
        self.dx_lim = 0.7
        self.n_steps_max = 150
        self.k_mom = 0.1
        self.kick = 50


    def forwardPass_fast(self, u_mat, x0, cost_fun_list):
        
        xi = torch.clone(x0)
        x_mat = torch.zeros((self.x_dim, self.n_points), device = self.device)
        
        cost = torch.tensor([0.])
        for ii in range(self.n_points):
            
            A_i1_i = self.A_fun(u_mat[:,ii], self.device)
            xi = A_i1_i @ xi
            x_mat[:,ii] = xi
            
            cost += cost_fun_list[ii](xi).to('cpu')
            
        return x_mat.to('cpu'), cost
    
    # for testing only
    def finite_diff_jacobian(self, u_mat, x0, cost_fun_list):
        # wrt single angVec index
        du = 1e-7
        dJ_du = torch.zeros(self.nu_opt)
        x_opt = u_mat.T.reshape(self.nu_opt)
        
        _, _, J0 = self.forwardPass_fast(u_mat, x0, cost_fun_list)
        
        for state in range(self.nu_opt):
            
            x_perturb = torch.clone(x_opt)
            x_perturb[state] += du
            u_mat_p = x_perturb.reshape(self.n_points, self.u_dim).T
            
            _, _, J_pos = self.forwardPass_fast(u_mat_p, x0, cost_fun_list)
            dJ_du[state] = (J_pos - J0) / du
    
        return dJ_du    

    def forwardPass(self, u_mat, x0, A_fun, A_jac_fun, c_grad_fun_list):
        
        x_dim = self.x_dim
        u_dim = self.u_dim
        nps = self.n_points
        
        # construct rotations from w
        x_mat = torch.zeros((x_dim, nps), device = self.device)
        
        
        # first step
        ii = 0
        A_i1_i = A_fun(u_mat[:,ii], self.device)
        x_mat[:,ii] = A_i1_i @ x0
        self.A_big[ii*x_dim:(ii+1)*x_dim, :] *= 0 
        
        #u_ave_mat = 0.5 * (u_mat[:,:-1] + u_mat[:,1:])
        for ii in range(1,self.n_points):
            tic = time.perf_counter()
            A_i1_i = A_fun(u_mat[:,ii], self.device)
            #A_k1_k_tensor[ii,:,:] = A_i1_i

            # propagate A and x
            #A_k_0_tensor[ii,:,:] = A_i1_i @ A_k_0_tensor[ii-1,:,:]
            x_mat[:,ii] = A_i1_i @ x_mat[:,ii-1]
            
            self.A_big[ii*x_dim:(ii+1)*x_dim, :] = A_i1_i @ self.A_big[(ii-1)*x_dim:ii*x_dim, :]
                
            self.A_big[ii*x_dim:(ii+1)*x_dim, ii*x_dim:(ii+1)*x_dim] = torch.eye(x_dim)
                    
            self.dx_du_tensor[ii,:,:] = A_jac_fun(u_mat[:,ii], x_mat[:,ii], self.device)

            
            self.dc_dx_vec[ii*x_dim:(ii+1)*x_dim] = c_grad_fun_list[ii](x_mat[:,ii])
            

        l_vec = self.dc_dx_vec[None,:] @ self.A_big
        
        dj_du = l_vec.reshape(nps, 1, x_dim) @ self.dx_du_tensor
        dj_du_reshaped = dj_du.reshape(u_dim * nps)
            
        return dj_du_reshaped.to('cpu')


    def optimize_trajectory(self, u_mat0, x0_in, cost_fun_list, c_grad_fun_list):
        
        x0 = x0_in.to(self.device)
        
        u_mat_full = torch.clone(u_mat0)

        # optimization vector
        x_opt = u_mat_full.T.reshape(self.nu)

        g_list = torch.zeros(self.n_steps_max)
        u_buf = torch.zeros(self.u_dim)
        d_step_prev = 0
        grad_f_prev = 0
        Hinv = self.Hinv
        
        # main optimizer loop
        for step in range(self.n_steps_max):
            
            # compute trajectory forward 
            dj_du = self.forwardPass(u_mat_full[:,1:], x0, self.A_fun, self.A_jacobian, c_grad_fun_list)
            
            # double check gradient dj_du with finite differences
            # dj_du2 = self.finite_diff_jacobian(u_mat_full[:,1:], x0, cost_fun_list)
            # print(dj_du)
            # print(dj_du2)
            # print(dj_du - dj_du2)
            
            # gradient of control vector penalty
            grad_f1 = self.sigma_u * x_opt[self.id_u1]
            
            grad_f2 = self.sigma_a * ( x_opt[self.id_u1] - 2*torch.concat((u_buf, x_opt[self.id_u1[0:-self.u_dim]])) 
                   - 2*torch.concat((x_opt[self.id_u1[self.u_dim:]], u_buf)) )
            
            grad_fu = grad_f1 + grad_f2
            
            grad_f = dj_du + grad_fu[self.id_u2]
            
            # update hessian 
            k = self.k2 if step > self.kick else self.k1
            if step > self.kick:
            #if False:
                s = d_step_prev
                y = grad_f - grad_f_prev
                I1 = self.I - s[:,None] @ y[None,:] / (y[None,:] @ s[:,None])
                I2 = self.I - y[:,None] @ s[None,:] / (y[None,:] @ s[:,None])
                I3 = s[:,None] @ s[None,:] / (y[None,:] @ s[:,None])
                Hinv = I1 @ Hinv @ I2 + I3
                
                dx = -Hinv @ grad_f
            else:
                dx = -grad_f
                                
            grad_f_prev = grad_f
            
            d_step  = k * dx
 
            # limit step length
            tmp   = torch.max(torch.abs(dx)) / self.dx_lim
            k_lim = tmp if tmp > 1 else 1
            d_step /= k_lim
            
            d_step  = self.k_mom * d_step_prev  + (1-self.k_mom) * d_step
            d_step_prev = d_step
                     
            x_opt[self.id_u2]  += d_step
            
            # unpack optimizer state
            u_mat_full = x_opt.reshape(self.n_points+1, self.u_dim).T
            
            g_norm = torch.linalg.norm(grad_f)
            g_list[step] = g_norm

            # exit early if converged
            if (g_norm < self.g_norm_lim):
                break

        # construct final outputs from optimizer solution
        self.u_mat  = u_mat_full
        x_mat, _ = self.forwardPass_fast(u_mat_full[:,1:], x0, cost_fun_list)
        self.x_mat = torch.concat((x0_in[:,None], x_mat), dim=1)

        accelMat  = (self.u_mat[:,1:] - self.u_mat[:,:-1])
        self.aMat = torch.concat((accelMat, accelMat[:,-1][:,None]), dim = 1)

        return self.x_mat, self.u_mat, g_list, step  
   
#------------------------------------------------------------------------------
# toy example problem functions
def A_fun(u, device = 'cpu'):
    return torch.tensor([ [1,  0,  u[0], 0],
                          [0,  1,  0,    u[1]], 
                          [0,  0,  1,    0  ],
                          [0,  0,  0,    1  ]], device = device)

def A_jacobian(u, x, device = 'cpu'):
    return torch.tensor([[ x[2], 0   ],
                         [ 0,    x[3]],
                         [ 0,    0   ],
                         [ 0,    0   ]], device = device)

def cost_penalty(rand_locs, x):
    
    sigma_obs = 0.7
    obstacle_loc = rand_locs
    n_obs = obstacle_loc.shape[0]
    obstacle = 0
    
    for obs in range(n_obs):
    
        obstacle_err = x[0:2] - obstacle_loc[obs,:]
        obstacle += 0.5 * torch.exp(-sigma_obs * obstacle_err @ obstacle_err.T)

    return  obstacle

def cost_goal(x):
    
    r_target = torch.tensor([10, 10])
    sink_err = x[0:2] - r_target
    
    sink1 =  100 * sink_err @ sink_err.T
    sink2 = -1e5 * torch.exp(-1 * sink_err @ sink_err.T)
    offset = 0
    
    return  sink1 + sink2 + offset

def cost_null(x):
    return torch.zeros(1)

def cost_gradient_null(x):
    return torch.zeros(4)


def cost3_function_gradient(r_target, R, x):
    tmp = R @ (x[0:2] - r_target)
    return torch.concat( (tmp, torch.zeros(2)) )


#------------------------------------------------------------------------------
class KernelCost2d():
    def __init__(self, w, h, n_grid, scale, noise, state_id, state_dim, device = 'cpu'):
        
        self.state_dim = state_dim
        self.state_id  = state_id
        self.device = device
        
        self.w = w
        self.h = h
        self.n_grid = n_grid
        self.scale = scale
        self.n_feat = n_grid**2
        x_vec = torch.linspace(-w/2, w/2, n_grid)
        y_vec = torch.linspace(-h/2, h/2, n_grid)
        
        self.X_grid, self.Y_grid = torch.meshgrid(x_vec, y_vec)
        
        self.x_locs = torch.stack((self.X_grid.flatten(), self.Y_grid.flatten()), dim = 0).to(device)
        
        self.Phi = self.create_kernel_matrix(self.x_locs).to(device)
        
        self.K_inv = torch.linalg.inv(self.Phi.T @ self.Phi + noise * torch.eye(self.n_feat, device = device))
        
        self.theta_vec = torch.zeros(self.n_feat, device = device)
        
    def compute_dc_dx(self, x_in):
        
        x = x_in[self.state_id]
        dc_dx = torch.zeros(self.state_dim, device = self.device)
        
        X = x[:,None] - self.x_locs
        
        distance_squared = torch.sum(X**2, dim = 0)
        f_vec = torch.exp(-0.5 * self.scale * distance_squared)
        
        F_mat = -(X * f_vec[None,:]) * self.scale
       
        dc_dx[self.state_id] = F_mat @ self.theta_vec
        
        return dc_dx
    
        
    def compute_cost(self, x):
        return self.create_feature_vector(x[self.state_id]) @ self.theta_vec
        
    def map_cost_function(self, cost_fun):
        
        cost_vec = torch.zeros(self.n_feat)
        
        for loc in range(self.n_feat):
            cost_vec[loc] = cost_fun(self.x_locs[:,loc].to('cpu'))
        
        self.theta_vec = self.K_inv @ (self.Phi.T @ cost_vec.to(self.device))
        #self.theta_vec[505] = 0.01
        
        
    def create_feature_matrix(self, x):
        distance_mat = torch.cdist(self.x_locs.T, x[None,:],  p=2)
        kernel_mat = torch.exp(-0.5 * self.scale * distance_mat.T**2)
        
        return kernel_mat
    
    def create_feature_vector(self, x):
        distance_mat = torch.cdist(self.x_locs.T, x[None,:],  p=2)
        
        X = x[:,None] - self.x_locs
        distance_squared = torch.sum(X**2, dim = 0)
        f_vec = torch.exp(-0.5 * self.scale * distance_squared)
        
        return f_vec
        
    def create_kernel_matrix(self, x_locs):

        distance_mat = torch.cdist(x_locs.T, x_locs.T, p=2)
        kernel_mat = torch.exp(-0.5 * self.scale * distance_mat**2)
        
        return kernel_mat
    
    def create_feature_derivative_matrix(self, x):
        
        f_vec = self.create_feature_vector(x)
        X = x[:,None] - self.x_locs
        F_mat = -(X * f_vec[None,:]) * self.scale
        
        return F_mat
    
    def plot_cost_surface(self, cost_fun, n_grid = 50):
    
        # construct fine grid
        x_vec = torch.linspace(-self.w/2, self.w/2, n_grid)
        y_vec = torch.linspace(-self.h/2, self.h/2, n_grid)
        
        X, Y = torch.meshgrid(x_vec, y_vec)
        
        Z  = torch.zeros((n_grid, n_grid))
        Z2 = torch.zeros((n_grid, n_grid))
        Zx = torch.zeros((n_grid, n_grid))
        Zy = torch.zeros((n_grid, n_grid))
        
        self.theta_vec = self.theta_vec.to('cpu')
        self.x_locs = self.x_locs.to('cpu')
        
        for ii in range(n_grid):
            for jj in range(n_grid):
                x = torch.tensor([x_vec[ii], y_vec[jj]])
                Z[ii,jj] = cost_fun(x)
                
                Z2[ii,jj] = self.create_feature_matrix(x) @ self.theta_vec
                
                grad = self.create_feature_derivative_matrix(x) @ self.theta_vec
                Zx[ii,jj] = grad[0]
                Zy[ii,jj] = grad[1]
                
        X = X.numpy()
        Y = Y.numpy()
        Z = Z.numpy()
        Z2 = Z2.numpy()
        Zx = Zx.numpy() 
        Zy = Zy.numpy() 
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.contourf(X, Y, Z, levels = 100)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('true cost')
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.contourf(X, Y, Z2, levels = 100)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('kernel aprox')
        
        fig, ax_2d = plt.subplots(1, 1)
        ax_2d.quiver(X.flatten(), Y.flatten(), Zx.flatten(), Zy.flatten())       
        
        return ax_2d, X, Y, Z2
                         
                         
#------------------------------------------------------------------------------
if __name__ == '__main__':
    plt.close('all')
    torch.manual_seed(3)

    x_dim    = 4
    u_dim    = 2
    n_points = 120
    
    x_plot_id = [0, 1]
    pos_start = [-15., -15.]
    x0 = torch.tensor([pos_start[0], pos_start[1], 1., 1.])
    
    speed = 0.0
    ang = 70 * np.pi/180
    u0 = torch.tensor([ np.cos(ang), np.sin(ang)]) * speed
    
    # attach boundary conditions to u trajectory
    u_mat = torch.zeros((u_dim, n_points+1))
    u_mat[:,:] = u0[:,None]
    u_mat[:,0] = u0
    
    a_scale = 0.33
    a_scale = 0.15
    a_scale = 0.18
    sigma_u = torch.tensor([1., 1.]) *4
    sigma_a = torch.tensor([1., 1.]) * a_scale
    
    r_target = torch.tensor([10., 10.])
    R        = torch.tensor([[0.9, 0.],  [0.,  0.9]]) * 0.001
    
    # kernel cost
    n_grid = 64
    w = 40
    h = 40
    kernel_scale = 0.01
    kernel_scale = 0.05
    noise = 1
    state_id = [0, 1]
    
    kernel_penalty = KernelCost2d(w, h, n_grid, 0.5, noise, state_id, x_dim, device)
    kernel_goal = KernelCost2d(w, h, n_grid, 0.05, noise, state_id, x_dim, device)
    
    rand_locs = torch.randn(30,2) * 10
    cost_penalty = partial(cost_penalty, rand_locs)

    kernel_penalty.map_cost_function(cost_penalty)
    kernel_goal.map_cost_function(cost_goal)
    
    c2_fun = kernel_penalty.compute_cost
    c2_grad_fun = kernel_penalty.compute_dc_dx
    
    c1_fun = kernel_goal.compute_cost
    c1_grad_fun = kernel_goal.compute_dc_dx
    
    cost_fun_list = list()
    c_grad_fun_list = list()
    for p in range(n_points):
        
        if p < (n_points - 5):
            c_grad_fun_list.append(c2_grad_fun)
            cost_fun_list.append(c2_fun)           
        else:
            c_grad_fun_list.append(c1_grad_fun)
            cost_fun_list.append(c1_fun)            
                
            
    traj = trajectoryOptimizer(sigma_u, sigma_a, x_dim, u_dim, n_points, A_fun, A_jacobian, device)
    tic = time.perf_counter()
    x_mat, u_mat, g_list, step  = traj.optimize_trajectory(u_mat, x0, cost_fun_list, c_grad_fun_list)
    timer = time.perf_counter() - tic
    print(f'timer: {timer}')
    
    x_plot = np.squeeze(x_mat[x_plot_id,:].numpy())
    
    ax_2d, X, Y, Z = kernel_goal.plot_cost_surface(cost_goal)
    
    ax_2d, X, Y, Z = kernel_penalty.plot_cost_surface(cost_penalty)
    
    plt.figure()
    plt.plot(x_plot.T)
    plt.title('x')

    u_mat_np = np.squeeze(u_mat.numpy())
    speed = np.sqrt(u_mat_np[0,:]**2 + u_mat_np[1,:]**2)
    
    accel = traj.aMat.numpy()
    
    u_dir = u_mat_np / speed
    u_lat = np.array([[0, 1], [-1, 0]]) @ u_dir

    accel_axial = np.sum(accel * u_dir, axis = 0)
    accel_radial = np.sum(accel * u_lat, axis = 0)
    
    plt.figure()
    plt.plot(u_mat_np.T)
    plt.title('vel')
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(speed)
    ax[0].set_title('speed')
    ax[0].grid('on')
    
    ax[1].plot(accel_axial)
    ax[1].plot(accel_radial)
    ax[1].set_title('acceleration')
    ax[1].legend(['axial','radial'])
    ax[1].set_xlabel('time')
    ax[1].grid('on')
    
    plt.figure()
    plt.plot(g_list)
    plt.title('optimizer gradient norm')
    
    plt.figure()
    plt.plot(x_plot[0,:], x_plot[1,:])
    plt.plot(pos_start[0], pos_start[1], 'go')
    plt.plot(r_target[0], r_target[1], 'ro')
    plt.legend(['trajectory','start', 'finish'])
    plt.contour(X, Y, Z, levels = 5)
    
    
    
    
    #--------------------------------------------------------------------------
    # plotting
    
    

    

    
        
        